"""Smoke / rubric-REJECTION battery for SliderGauntletScene — NullRobot probes.

solve.py is the acceptance proof (the rubric accepts a real dragged trajectory and the
latched credit is monotone along it). This battery proves the rubric REJECTS wrong
outcomes, and that the three physical interlocks the task rests on are load-bearing
geometry, not fiat:

  - CAPTIVITY: the roof, not a rubric clause, forbids lifting any piece out;
  - ORDER:     the sealing blocker, not a latch, stops the runner at its station;
  - THE PLUG:  the red plug, not a script, dead-ends the wrong side-corridor arm.

Force probes follow the non-vacuity rule: every "it was refused" assertion is paired
with a readback that the force actually acted (the piece measurably moved / pressed).
Constructed states (teleports) are used only to build settled WRONG outcomes for the
rubric to refuse — no probe here reaches success().

Checks:
   1. settle          — seeded reset settles finite; both stations sealed; authored
                        masses took (get_masses readback); score ~0, no success;
   2. randomization   — 8 seeded resets: >= 3 of the 4 open-side layouts appear, the
                        plug PHYSICALLY fills the closed arm on every draw, runner
                        start depth and board yaw/xy spread;
   3. null-policy     — 2.5 s of nothing: score ~0, no success, runner stays put;
   4. CAPTIVITY       — 4x-weight straight-up pull on the runner: it presses into the
                        roof (z RISES — the pull provably acts) but the roof caps it
                        (never near escaping); released, it drops back; no credit;
   5. ORDER interlock — hard forward shove on the runner with both stations sealed:
                        it MOVES (>= 15 mm) then stalls against blocker A, never
                        passing the station; it cannot climb over (roof); tiny
                        approach credit only, no success;
   6. PLUG interlock  — blocker A shoved toward its red plug: it MOVES then jams
                        within ~2 cm of the axis — the station stays sealed, no clear
                        credit ever latches;
   7. near-miss       — both stations legitimately cleared, the runner dragged to the
                        exit LIP but not out: score capped at 0.65, no success;
   8. wrong object    — a BLOCKER teleported onto the tray floor, the runner still in
                        the corridor: the tray clause keys on the runner — no success;
   9. beside the tray — the runner CONSTRUCTED on the ground beside the tray (y gate)
                        and beyond the end wall (x gate): both refused;
  10. clear latch     — blocker A cleared (0.20 credit), then pushed BACK to re-seal
                        its station: the station reads obstructed again but the
                        latched credit survives (score stays >= 0.20), no success;
  11. success+revoke  — full inline solve: success() and score exactly 1.0, still
                        true 1 s later; then the runner is plucked back into the
                        corridor (constructed): success revokes LIVE, score falls
                        back to the latched 0.65 — 1.0 iff success;
  12. frames.npz      — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -u -m simgen_tasks.robosuite_env_i223.smoke --headless
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
    from . import scene as task_scene  # noqa: F401 — import registers the scene + env
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    rl = scene.runner_loc()[0]
    bl = scene.blocker_loc()[0]
    print(f"[smoke] {tag:16s} | runner=({float(rl[0]):+.3f},{float(rl[1]):+.3f},"
          f"{float(rl[2]):+.3f}) blkA=({float(bl[0, 0]):+.3f},{float(bl[0, 1]):+.3f}) "
          f"blkB=({float(bl[1, 0]):+.3f},{float(bl[1, 1]):+.3f}) "
          f"obst={scene.obstructing()[0].tolist()} clear={scene._clear[0].tolist()} "
          f"prog={float(scene._prog[0]):.3f} in_tray={bool(scene.in_tray()[0])} "
          f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.slider_gauntlet")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    zero = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.20, -0.75, 0.75)) + o),
                                tuple(np.array((0.40, 0.00, 0.05)) + o),
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

    def board_dir(loc_xyz) -> torch.Tensor:
        """Board-local direction -> world (n, 3). The board is kinematic and never
        moves after reset, so one evaluation per push is exact."""
        d = torch.tensor(loc_xyz, device=device, dtype=torch.float32).expand(n, 3)
        return quat_apply(scene.board.data.root_quat_w, d)

    def push(body, dir_w: torch.Tensor, mag: float, steps: int) -> None:
        """Constant world-frame force at the CoM for `steps` substeps, then clear."""
        f = (mag * dir_w).unsqueeze(1)
        for _ in range(steps):
            body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
            _step(1)
        body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())

    def servo(body, dir_loc, stop_fn, *, kp: float, v_des: float = 0.12,
              fmax: float = 2.5, max_steps: int = 1600) -> bool:
        """The solve's velocity-servo drag (raw world frame — proven on this pod)."""
        d_w = board_dir(dir_loc)
        for _ in range(max_steps):
            v_along = (body.data.root_lin_vel_w * d_w).sum(dim=-1)
            f = (d_w * (kp * (v_des - v_along)).clamp(-fmax, fmax).unsqueeze(-1))
            body.set_external_force_and_torque(f.unsqueeze(1), zero,
                                               env_ids=_all_ids(), is_global=True)
            _step(1)
            if stop_fn():
                body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
                return True
        body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        return False

    def write_local(body, loc_xyz) -> None:
        """Teleport a body to a board-local pose (board yaw), zero velocity."""
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float32).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.board.data.root_pos_w + quat_apply(
            scene.board.data.root_quat_w, loc)
        st[:, 3:7] = scene.board.data.root_quat_w
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def clear_station(i: int) -> bool:
        side = float(scene.open_side[0, i])
        body = scene.blockers[scene.BLOCKER_NAMES[i]]

        def done():
            return float(scene.blocker_loc()[0, i, 1]) * side >= 0.095

        ok = servo(body, (0.0, side, 0.0), done, kp=9.0)
        _step(60)  # streak latch
        return ok

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    # ================= 1. settle + authored masses ================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    m_run = float(scene.runner.root_physx_view.get_masses().flatten()[0])
    m_blk = float(scene.blockers["blocker_a"].root_physx_view.get_masses().flatten()[0])
    print(f"[smoke] mass readback: runner {m_run:.3f} kg (want {c.runner_mass}), "
          f"blocker {m_blk:.3f} kg (want {c.blocker_mass})", flush=True)
    check("settle: seeded reset settles finite, both stations sealed, authored "
          "masses took, score ~0, no success",
          bool(torch.isfinite(scene.runner.data.root_pos_w).all())
          and bool(scene.obstructing()[0].all())
          and abs(m_run - c.runner_mass) < 0.02 and abs(m_blk - c.blocker_mass) < 0.02
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    combos: set = set()
    x0s: list[float] = []
    yaws: list[float] = []
    xys: list[torch.Tensor] = []
    plug_ok = True
    for s in range(8):
        torch.manual_seed(300 + s)
        env.reset()
        _step(5)
        _refresh()
        side = scene.open_side[0].tolist()
        combos.add((int(side[0]), int(side[1])))
        x0s.append(float(scene.runner_loc()[0, 0]))
        yaws.append(yaw_of(scene.board.data.root_quat_w[0]))
        xys.append(scene.board.data.root_pos_w[0, :2].clone())
        for i, p in enumerate(scene.PLUG_NAMES):
            py = float(scene._board_local(scene.plugs[p].data.root_pos_w)[0, 1])
            plug_ok = plug_ok and (py * side[i] < -0.05)  # plug fills the CLOSED arm
    yaw_spread = max(yaws) - min(yaws)
    x0_spread = max(x0s) - min(x0s)
    xy_spread = float(max((a - b).norm() for a in xys for b in xys))
    print(f"[smoke] randomization: open-side combos {sorted(combos)}, "
          f"x0 spread {x0_spread * 1000:.0f}mm, yaw spread {yaw_spread:.1f}deg, "
          f"board xy spread {xy_spread * 1000:.0f}mm, plugs-in-closed-arm={plug_ok}",
          flush=True)
    check("randomization-is-real: >= 3 of 4 open-side layouts over 8 resets, plug "
          "physically fills the closed arm on every draw, runner depth / board "
          "yaw / board xy all spread",
          len(combos) >= 3 and plug_ok and x0_spread > 0.01
          and yaw_spread > 5.0 and xy_spread > 0.01)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    x_start = float(scene.runner_loc()[0, 0])
    _step(300)
    _report("null-policy")
    check("null-policy-fails: 2.5 s of nothing — score ~0, no success, runner stays put",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0])
          and abs(float(scene.runner_loc()[0, 0]) - x_start) < 0.005)

    # ================= 4. CAPTIVITY: the roof interlock ===========================================
    # 4x-weight straight-up pull (up is yaw-invariant, so no frame-drag caveat). The
    # runner must PRESS INTO the roof — z rises, proving the pull acts — yet stay far
    # below any escape (body centre would need z > roof underside to leave).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    z_rest = float(scene.runner_loc()[0, 2])
    up = torch.zeros(n, 3, device=device)
    up[:, 2] = 1.0
    z_peak = 0.0
    for _ in range(18):  # 18 x 10 = 180 substeps = 1.5 s of sustained pull
        push(scene.runner, up, 4.0 * c.runner_mass * 9.81, 10)
        z_peak = max(z_peak, float(scene.runner_loc()[0, 2]))
    _step(90)  # release: gravity returns it to the floor
    _report("captivity")
    z_back = float(scene.runner_loc()[0, 2])
    print(f"[smoke] captivity: z rest {z_rest * 1000:.1f}mm -> peak {z_peak * 1000:.1f}mm "
          f"(roof underside {c.wall_top * 1000:.0f}mm) -> released {z_back * 1000:.1f}mm",
          flush=True)
    check("CAPTIVITY: a 4x-weight straight-up pull presses the runner into the roof "
          "(z demonstrably rises) but cannot lift it out; released, it drops back; "
          "no credit",
          z_peak > z_rest + 0.004 and z_peak < c.wall_top
          and z_back < z_rest + 0.004
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 5. ORDER interlock: the sealed station stops the runner ====================
    # Hard forward shove (3x the drag friction, still under the tipping bound) with
    # both stations sealed: the runner must MOVE (the shove provably acts), stall
    # against blocker A short of the station, and stay under the roof (no climbing).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    x_before = float(scene.runner_loc()[0, 0])
    z_max = 0.0
    fwd = board_dir((1.0, 0.0, 0.0))
    for _ in range(30):  # 30 x 10 = 300 substeps = 2.5 s of sustained shove
        push(scene.runner, fwd, 2.0, 10)
        z_max = max(z_max, float(scene.runner_loc()[0, 2]))
    _step(60)
    _report("order-interlock")
    x_after = float(scene.runner_loc()[0, 0])
    print(f"[smoke] order shove: x {x_before * 1000:+.0f} -> {x_after * 1000:+.0f}mm "
          f"(blocker contact at -50mm, station at 0), z_max {z_max * 1000:.1f}mm",
          flush=True)
    check("ORDER interlock: a hard forward shove moves the runner (>= 15 mm) but it "
          "stalls against sealed station A — never passes, never climbs; tiny "
          "approach credit only, no success",
          x_after > x_before + 0.015 and x_after < -0.035 and z_max < 0.055
          and bool(scene.obstructing()[0].all())
          and float(scene.score()[0]) <= 0.10 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. PLUG interlock: the wrong arm dead-ends =================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    side0 = float(scene.open_side[0, 0])
    y_before = float(scene.blocker_loc()[0, 0, 1])
    toward_plug = board_dir((0.0, -side0, 0.0))
    for _ in range(24):  # 2 s of sustained shove toward the plug
        push(scene.blockers["blocker_a"], toward_plug, 2.0, 10)
    _step(60)
    _report("plug-interlock")
    y_after = float(scene.blocker_loc()[0, 0, 1])
    moved = (y_after - y_before) * (-side0)
    print(f"[smoke] plug shove: blocker A y {y_before * 1000:+.1f} -> "
          f"{y_after * 1000:+.1f}mm (moved {moved * 1000:.1f}mm toward the plug; jam "
          f"expected ~6mm off-axis)", flush=True)
    check("PLUG interlock: blocker A shoved toward its red plug moves (>= 3 mm) then "
          "jams within 2 cm of the axis — the station stays sealed, no clear credit",
          moved > 0.003 and abs(y_after) < 0.020
          and bool(scene.obstructing()[0, 0]) and not bool(scene._clear[0, 0])
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 7. near-miss: runner at the exit lip, not out ==============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    ok_a = clear_station(0)
    ok_b = clear_station(1)

    def at_lip():
        return float(scene.runner_loc()[0, 0]) > 0.165

    ok_run = servo(scene.runner, (1.0, 0.0, 0.0), at_lip, kp=12.0)
    _step(120)
    _report("near-miss")
    x_lip = float(scene.runner_loc()[0, 0])
    s_lip = float(scene.score()[0])
    check("near-miss: both stations cleared and the runner dragged to the exit lip "
          "(but not out) — full latched credit yet score capped at 0.65, no success",
          ok_a and ok_b and ok_run and 0.165 <= x_lip < 0.195
          and not bool(scene.in_tray()[0]) and not bool(scene.success()[0])
          and 0.50 <= s_lip <= 0.6500002)

    # ================= 8. wrong object in the tray ================================================
    # CONSTRUCT blocker B resting on the tray floor (a state no policy can reach —
    # which is the point): the tray clause keys on the RUNNER.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_local(scene.blockers["blocker_b"], (0.27, 0.0, 0.016))
    _step(120)
    _report("wrong-object")
    bl = scene.blocker_loc()[0, 1]
    check("negative (wrong object): a BLOCKER rests on the tray floor, the runner "
          "still in the corridor — no success",
          float(bl[0]) > c.tray_x0 and float(bl[2]) < c.tray_z
          and not bool(scene.in_tray()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.6500002)

    # ================= 9. beside / beyond the tray ================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_local(scene.runner, (0.27, 0.125, 0.016))  # on the ground BESIDE the tray wall
    _step(120)
    _report("beside-tray")
    beside_refused = (not bool(scene.in_tray()[0])) and not bool(scene.success()[0])
    y_beside = float(scene.runner_loc()[0, 1])
    write_local(scene.runner, (0.40, 0.0, 0.016))    # on the ground BEYOND the end wall
    _step(120)
    _report("beyond-tray")
    beyond_refused = (not bool(scene.in_tray()[0])) and not bool(scene.success()[0])
    x_beyond = float(scene.runner_loc()[0, 0])
    check("negative (tray gates): the runner at ground level BESIDE the tray "
          f"(y={y_beside * 1000:+.0f}mm) and BEYOND its end wall "
          f"(x={x_beyond * 1000:+.0f}mm) — both refused",
          beside_refused and beyond_refused
          and abs(y_beside) > c.tray_y and x_beyond > c.tray_x1)

    # ================= 10. the clear latch survives re-sealing ====================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ok_a = clear_station(0)
    s_clear = float(scene.score()[0])

    def resealed():
        return abs(float(scene.blocker_loc()[0, 0, 1])) < 0.010

    side0 = float(scene.open_side[0, 0])
    ok_back = servo(scene.blockers["blocker_a"], (0.0, -side0, 0.0), resealed, kp=9.0)
    _step(60)
    _report("latch-reseal")
    check("clear latch: blocker A cleared (score 0.20) then pushed BACK to re-seal "
          "the station — the live state reads obstructed again but the latched "
          "credit survives; still no success",
          ok_a and s_clear >= c.w_clear - 1e-6 and ok_back
          and bool(scene.obstructing()[0, 0]) and bool(scene._clear[0, 0])
          and float(scene.score()[0]) >= c.w_clear - 1e-6
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 11. success is live: full solve, then revocation ===========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    ok_a = clear_station(0)
    ok_b = clear_station(1)

    def out_or_dropping():
        loc = scene.runner_loc()[0]
        return float(loc[0]) > 0.215 or float(loc[2]) < 0.038

    ok_run = servo(scene.runner, (1.0, 0.0, 0.0), out_or_dropping, kp=12.0)
    _step(300)
    _report("success")
    got = bool(scene.success()[0]) and float(scene.score()[0]) == 1.0
    _step(120)  # 1 s hands-off: still true
    held = bool(scene.success()[0]) and float(scene.score()[0]) == 1.0
    # revocation: pluck the runner back into the (empty) corridor — constructed state
    write_local(scene.runner, (0.15, 0.0, 0.046))
    _step(60)
    _report("revoked")
    s_rev = float(scene.score()[0])
    check("success-is-live: the dragged solve reaches success() with score exactly "
          "1.0 (held 1 s); plucking the runner back out revokes success LIVE and "
          "the score falls to the latched 0.65 — 1.0 iff success",
          ok_a and ok_b and ok_run and got and held
          and not bool(scene.success()[0]) and 0.64 <= s_rev <= 0.6500002)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.slider_gauntlet")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) >= 20)

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
    except BaseException:  # noqa: BLE001 — die loudly, never idle until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

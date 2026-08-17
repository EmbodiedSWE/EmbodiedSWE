"""Smoke / rubric-REJECTION battery for RackTrayServeScene — NullRobot, teleported probes.

solve.py is the acceptance proof (force-slide the tray out -> gravity-seat it in the
well -> drop the front bowl on it). This battery proves the rubric REJECTS wrong
outcomes — above all that the seed's own strategy (put the front bowl at the goal,
one pick-and-place into the waiting receptacle) scores ~0, that the tray's captivity
is REAL physics (an upward pull 3x its weight cannot lift it out of the rack while
the same force freely slides it along its one open DoF), and that the forced order
is enforced by contact (a tray dropped onto an occupied well rests high and is
refused). Every probe is CONSTRUCTED (teleport, real physics steps, judge) —
instrumentation, never a solution.

Checks:
  1. settle/no-NaN    — seeded reset settles finite: tray at its latched stow depth,
                        masses authored (tray 0.15, bowls 3 x 0.10), score ~0;
  2. randomization    — over 8 seeded resets the stand yaw takes >= 4 distinct
                        values, the target BODY >= 2 values, xy jitter and the
                        stow depth differ between resets (readback);
  3. null-policy      — 240 idle steps: no stage latches, score ~0, no success;
  4. captivity        — a 4.5 N upward pull (3x the tray's weight) on the stowed
                        tray: the roof rail stops it (peak rise < 20 mm), the tray
                        stays in the rack, no out credit;
  5. free-DoF contrast— the SAME 4.5 N applied along the channel slides the tray
                        >= 30 mm: the captivity result is the rail, not a freeze;
  6. SEED STRATEGY    — "put the front bowl at the goal": the front bowl set down
                        into the BARE well (the literal seed plan). It rests in the
                        well but success is False and score stays ~0 — the goal
                        surface was never prepared;
  7. blocked install  — continuing 6: the tray dropped onto the OCCUPIED well rests
                        on the bowl >= 20 mm above the seat band, tilt-prone, never
                        seated — the physical order-forcer;
  8. wrong place      — tray flat on the open counter + the front bowl centered on
                        it: the bowl-on-tray geometry is satisfied, but the tray is
                        not seated in the well — no seat/load credit, no success;
  9. rim-perch        — a 45-degree-misyawed tray dropped over the well (diagonal
                        240 mm > 190 mm opening) rests ON the rim >= 8 mm above the
                        seat band: not seated;
 10. offset near-miss — on a properly seated tray, the front bowl set down 65 mm
                        off-center (outside the 55 mm gate): on_tray False, no
                        success, score caps at the seat stage;
 11. wrong bowl       — a DECOY bowl centered perfectly on the seated tray: only
                        the front bowl counts — no load credit, no success;
 12. contamination    — the TARGET placed correctly AND a decoy also on the tray:
                        decoys_clear False kills success even though every other
                        clause holds (score caps at the loaded stage);
 13. frames.npz       — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_WD = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_WD.daemon = True
_WD.start()

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
    c = scene.cfg
    _refresh()
    t = int(scene.target[0])
    lx = float(scene.tray_local_x()[0]) * 1000
    dz = (float(scene.tray.data.root_pos_w[0, 2] - scene.stand.data.root_pos_w[0, 2])
          - (c.base_h + c.tray_thick / 2)) * 1000
    print(f"[smoke] {tag:22s} | tgt={t} tray_x={lx:+7.1f}mm seat_dz={dz:+6.1f}mm "
          f"seated={bool(scene.tray_seated()[0])} "
          f"on_tray={scene.on_tray()[0].tolist()} clear={bool(scene.decoys_clear()[0])} "
          f"latch=(O{int(scene._s_out[0])},S{int(scene._s_seated[0])},"
          f"L{int(scene._s_loaded[0])}) success={bool(scene.success()[0])} "
          f"score={float(scene.score()[0]):.3f} frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    from isaaclab.utils.math import quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rack_tray_serve")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.80, 1.10)) + o),
                                tuple(np.array((0.03, 0.03, 0.44)) + o),
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

    xs = torch.tensor(c.slot_xs, device=device)
    r_low = torch.tensor([0.0, 0.0, -0.080], device=device)
    zero3 = torch.zeros(n, 1, 3, device=device)

    def rack_yaw() -> float:
        _refresh()
        return float(scene._yaw_of(scene.rack.data.root_quat_w)[0])

    def stand_yaw() -> float:
        _refresh()
        return float(scene._yaw_of(scene.stand.data.root_quat_w)[0])

    def body_in_slot(s: int) -> int:
        _refresh()
        slots = [int((xs - (scene.bowls[i].data.root_pos_w[0, 0]
                            - scene.env_origins[0, 0])).abs().argmin())
                 for i in range(3)]
        return slots.index(s)

    def tray_write(pos_w: torch.Tensor, yaw: float) -> None:
        """Teleport the tray FLAT at `yaw` (zero velocity) to a world point."""
        _refresh()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        scene.tray.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def bowl_write(i: int, pos_w: torch.Tensor, yaw: float = 0.0) -> None:
        """Teleport bowl i (upright at `yaw`, zero velocity, bottom-center origin)."""
        _refresh()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        scene.bowls[i].write_root_state_to_sim(st, _all_ids())
        _refresh()

    def settle(max_steps: int = 300) -> None:
        for _ in range(max_steps):
            _step(1)
            _refresh()
            if int(scene._still_tray[0]) >= c.still_steps:
                break
        _step(30)
        _refresh()

    def install_tray(yaw_off: float = 0.0) -> None:
        """Solve phase 2 as a probe: teleport the tray 30 mm above the seat, flat,
        yaw = stand yaw + `yaw_off`, then a hands-off gravity drop + settle."""
        _refresh()
        entry = scene.stand.data.root_pos_w.clone()
        entry[:, 2] += c.base_h + c.tray_thick / 2 + 0.030
        tray_write(entry, yaw=stand_yaw() + yaw_off)
        settle()

    def drop_bowl_at(i: int, off_local: tuple[float, float], dz: float = 0.025) -> None:
        """Drop bowl i from `dz` above the tray top, offset `off_local` (m) from the
        tray center along the STAND's local axes; gravity does the placement."""
        _refresh()
        sy = stand_yaw()
        cos_s, sin_s = math.cos(sy), math.sin(sy)
        drop = scene.tray.data.root_pos_w.clone()
        drop[:, 0] += off_local[0] * cos_s - off_local[1] * sin_s
        drop[:, 1] += off_local[0] * sin_s + off_local[1] * cos_s
        drop[:, 2] += c.tray_thick / 2 + dz
        bowl_write(i, drop, yaw=0.0)
        for _ in range(300):
            _step(1)
            _refresh()
            if int(scene._still_bowl[0, i]) >= c.still_steps:
                break
        _step(30)
        _refresh()

    def zero_tray_wrench() -> None:
        scene.tray.set_external_force_and_torque(zero3, zero3)

    def pull_tray_up(newtons: float, max_steps: int, stop_rise: float) -> float:
        """Constant WORLD-up force on the tray (converted to its body frame each
        step); returns the PEAK rise (m) of its center, sampled with the force still
        on — gravity restores the state before any post-settle readback."""
        _refresh()
        z_start = float(scene.tray.data.root_pos_w[0, 2])
        f_w = torch.tensor([0.0, 0.0, newtons], device=device)
        forces = torch.zeros(n, 1, 3, device=device)
        peak = 0.0
        for _ in range(max_steps):
            _refresh()
            q = scene.tray.data.root_quat_w
            forces[:, 0, :] = quat_apply_inverse(q, f_w.expand(n, 3))
            scene.tray.set_external_force_and_torque(forces, zero3)
            _step(1)
            _refresh()
            peak = max(peak, float(scene.tray.data.root_pos_w[0, 2]) - z_start)
            if peak > stop_rise:
                break
        zero_tray_wrench()
        return peak

    def slide_tray(newtons: float, max_steps: int, stop_travel: float) -> float:
        """Velocity-capped WORLD force along the rack's +x with the low-push-point
        torque (solve phase 1's actuator); returns the settled channel travel (m)."""
        _refresh()
        ryaw = rack_yaw()
        ex = torch.tensor([math.cos(ryaw), math.sin(ryaw), 0.0], device=device)
        x0 = float(scene.tray_local_x()[0])
        forces = torch.zeros(n, 1, 3, device=device)
        torques = torch.zeros(n, 1, 3, device=device)
        for _ in range(max_steps):
            _refresh()
            if float(scene.tray_local_x()[0]) - x0 > stop_travel:
                break
            v_along = float((scene.tray.data.root_lin_vel_w[0] * ex).sum())
            if v_along < 0.15:
                f_w = newtons * ex
                t_w = torch.linalg.cross(r_low, f_w)
                q = scene.tray.data.root_quat_w
                forces[:, 0, :] = quat_apply_inverse(q, f_w.expand(n, 3))
                torques[:, 0, :] = quat_apply_inverse(q, t_w.expand(n, 3))
            else:
                forces[:, 0, :] = 0.0
                torques[:, 0, :] = 0.0
            scene.tray.set_external_force_and_torque(forces, torques)
            _step(1)
        zero_tray_wrench()
        _step(45)
        _refresh()
        return float(scene.tray_local_x()[0]) - x0

    def seat_dz() -> float:
        """Tray center height above the well-floor seat level (m)."""
        _refresh()
        return (float(scene.tray.data.root_pos_w[0, 2]
                      - scene.stand.data.root_pos_w[0, 2])
                - (c.base_h + c.tray_thick / 2))

    # ================= 1. settle / no-NaN + authoring readback ====================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    tray_mass = float(scene.tray.root_physx_view.get_masses().reshape(-1)[0])
    masses = [float(scene.bowls[i].root_physx_view.get_masses().reshape(-1)[0])
              for i in range(3)]
    print(f"[smoke] authored masses readback: tray={tray_mass:.3f} bowls={masses}",
          flush=True)
    stow_err = abs(float(scene.tray_local_x()[0]) - float(scene._stow_x[0]))
    check("settle/no-NaN: seeded reset settles finite — tray at its latched stow "
          "depth, masses authored (tray 0.15, bowls 3 x 0.10), score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.at_rest()[0])
          and abs(tray_mass - c.tray_mass) < 0.02
          and all(abs(mv - c.bowl_mass) < 0.02 for mv in masses)
          and stow_err < 0.015
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    yaws, targets = set(), set()
    jit, stows = [], []
    for s in range(8):
        torch.manual_seed(300 + 17 * s)
        env.reset()
        _step(10)
        _refresh()
        yaws.add(round(math.degrees(stand_yaw())))
        targets.add(int(scene.target[0]))
        stows.append(float(scene._stow_x[0]))
        jit.append(torch.cat([scene.stand.data.root_pos_w[0, :2],
                              scene.rack.data.root_pos_w[0, :2],
                              scene.bowls[0].data.root_pos_w[0, :2]]).clone())
    d_jit = max(float((a - b).abs().max()) for a in jit for b in jit)
    d_stow = max(stows) - min(stows)
    print(f"[smoke] randomization: stand_yaws={sorted(yaws)} targets={sorted(targets)} "
          f"max_jitter_delta={d_jit * 1000:.1f}mm stow_spread={d_stow * 1000:.1f}mm",
          flush=True)
    check("randomization-is-real: over 8 seeded resets the stand yaw takes >= 4 "
          "values, the target body >= 2 values, xy jitter and stow depth differ",
          len(yaws) >= 4 and len(targets) >= 2 and d_jit > 0.003 and d_stow > 0.002)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps — no stage latches, score ~0, no success",
          not bool(scene._s_out[0]) and not bool(scene._s_seated[0])
          and not bool(scene._s_loaded[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 4 + 5. captivity is real physics (and not a freeze) ========================
    # (a) a 4.5 N upward pull — 3x the tray's 1.47 N weight — on the STOWED tray:
    # its top edge jams on the rack's roof rail within a few mm. The seed's universal
    # primitive (lift the thing) is physically unavailable.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    rise = pull_tray_up(4.5, max_steps=180, stop_rise=0.05)
    _step(60)
    _report("captive-pull")
    lx_after = float(scene.tray_local_x()[0])
    print(f"[smoke] captivity: 4.5N up-pull peak rise = {rise * 1000:.1f}mm, "
          f"tray_x after = {lx_after * 1000:+.1f}mm", flush=True)
    check("captivity: a 4.5 N upward pull (3x weight) cannot lift the stowed tray "
          "past the roof rail (peak rise < 20 mm); it stays in the rack, no out credit",
          rise < 0.020 and lx_after < 0.0 and not bool(scene.tray_out()[0])
          and not bool(scene._s_out[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.15)
    # (b) the SAME 4.5 N along the channel slides the tray freely — the pull result
    # above is the rail blocking, not friction or a frozen body.
    travel = slide_tray(4.5, max_steps=240, stop_travel=0.05)
    _report("free-DoF-slide")
    print(f"[smoke] free-DoF contrast: 4.5N along-channel travel = "
          f"{travel * 1000:.1f}mm", flush=True)
    check("free-DoF-contrast: the SAME 4.5 N applied along the channel slides the "
          "tray >= 30 mm — the captivity above is the rail, not a freeze",
          travel > 0.030)

    # ================= 6. SEED STRATEGY: put the front bowl at the goal ===========================
    # The seed's whole plan is one pick-and-place of the front bowl onto the waiting
    # goal surface. Executed verbatim against THIS scene: the front bowl is set down
    # into the (bare) well of the serving stand. It lands fine — and is worth ~0,
    # because the serving surface was never prepared.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    _refresh()
    goal = scene.stand.data.root_pos_w.clone()
    goal[:, 2] += c.base_h + 0.030
    _REC["on"] = True
    bowl_write(front, goal, yaw=0.0)
    for _ in range(300):
        _step(1)
        _refresh()
        if int(scene._still_bowl[0, front]) >= c.still_steps:
            break
    _step(30)
    _report("seed-strategy")
    _REC["on"] = False
    _refresh()
    bowl_dz = float(scene.bowls[front].data.root_pos_w[0, 2]
                    - scene.stand.data.root_pos_w[0, 2])
    check("SEED STRATEGY: the front bowl set down into the BARE well (the literal "
          "seed plan) rests there — success False, score ~0: no stage was earned",
          abs(bowl_dz - c.base_h) < 0.008
          and not bool(scene._s_seated[0]) and not bool(scene._s_loaded[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02)

    # ================= 7. blocked install: the physical order-forcer ==============================
    # Continue: now try to install the tray anyway (solve's own aligned entry drop).
    # It lands ON the bowl, >= 20 mm above the seat band — never seated. The bowl
    # must come OUT before the tray can go in; the order is enforced by contact.
    _REC["on"] = True
    install_tray(yaw_off=0.0)
    _report("blocked-install")
    _REC["on"] = False
    dz7 = seat_dz()
    print(f"[smoke] blocked install: tray rests {dz7 * 1000:+.1f}mm above the seat",
          flush=True)
    check("blocked-install: the tray dropped onto the OCCUPIED well rests on the "
          "bowl >= 20 mm above the seat band — tray_seated False, no seat credit, "
          "no success",
          dz7 > 0.020 and not bool(scene.tray_seated()[0])
          and not bool(scene._s_seated[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_out + 1e-4)

    # ================= 8. right parts, wrong place ================================================
    # Tray flat on the open counter, front bowl centered on it: the bowl-relative-
    # to-tray geometry is FULLY satisfied — but the tray is not seated in the well.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    _refresh()
    spot = scene.env_origins.clone()
    spot[:, 0] += -0.25
    spot[:, 1] += 0.05
    spot[:, 2] = scene.stand.data.root_pos_w[:, 2] + c.tray_thick / 2 + 0.003
    tray_write(spot, yaw=0.0)
    settle()
    drop_bowl_at(front, (0.0, 0.0))
    _report("wrong-place")
    on_t = bool(scene._gather_tgt(scene.on_tray())[0])
    check("wrong-place: tray flat on the open counter with the front bowl centered "
          "on it — on_tray True yet seated False: no seat/load credit, no success",
          on_t and not bool(scene.tray_seated()[0]) and not bool(scene._s_seated[0])
          and not bool(scene._s_loaded[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_out + 1e-4)

    # ================= 9. rim-perch: misyawed tray cannot seat ====================================
    # 45-degree misyaw: the tray's 240 mm diagonal cannot enter the 190 mm opening
    # (entry needs alignment within ~8 deg); it rests ON the rim, well above the band.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    install_tray(yaw_off=math.pi / 4)
    _report("rim-perch")
    _REC["on"] = False
    dz9 = seat_dz()
    print(f"[smoke] rim-perch: tray rests {dz9 * 1000:+.1f}mm above the seat", flush=True)
    check("rim-perch: a 45-deg-misyawed tray (diagonal 240 mm > 190 mm opening) "
          "rests ON the rim >= 8 mm above the seat band — not seated, no success",
          dz9 > 0.008 and not bool(scene.tray_seated()[0])
          and not bool(scene._s_seated[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_out + 1e-4)

    # ================= 10. offset near-miss on a seated tray ======================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    install_tray(yaw_off=0.0)
    seated10 = bool(scene.tray_seated()[0])
    drop_bowl_at(front, (0.0, 0.065))  # 65 mm off-center: outside the 55 mm gate
    _report("offset-near-miss")
    _refresh()
    d_xy, _dz = scene._bowl_rel_tray()
    off = float(d_xy[0, front])
    check("offset-near-miss: on a properly seated tray the front bowl set down "
          "65 mm off-center (gate: 55 mm) — on_tray False, no load credit, no "
          "success, score caps at the seat stage",
          seated10 and 0.055 < off < 0.095
          and not bool(scene._gather_tgt(scene.on_tray())[0])
          and not bool(scene._s_loaded[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_seated + 1e-4)

    # ================= 11. wrong bowl centered on the seated tray =================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    decoy = body_in_slot(1)  # the MIDDLE bowl, not the front one
    install_tray(yaw_off=0.0)
    seated11 = bool(scene.tray_seated()[0])
    _REC["on"] = True
    drop_bowl_at(decoy, (0.0, 0.0))
    _report("wrong-bowl")
    _REC["on"] = False
    check("wrong-bowl: the MIDDLE bowl centered perfectly on the seated tray — "
          "only the front bowl counts: decoys_clear False, no load credit, no "
          "success, score caps at the seat stage",
          seated11 and bool(scene.on_tray()[0, decoy])
          and not bool(scene.decoys_clear()[0])
          and not bool(scene._s_loaded[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_seated + 1e-4)

    # ================= 12. contamination: correct target + intruding decoy ========================
    # The target is placed WITHIN tolerance and a decoy also ends up on the tray:
    # every clause but decoys_clear holds, the loaded stage latches — and success
    # still refuses. Offsets are along the stand's x so the two square bowls
    # (96 mm) clear each other (centers 102 mm apart).
    torch.manual_seed(100)
    env.reset()
    _step(60)
    front = body_in_slot(0)
    decoy = body_in_slot(1)
    install_tray(yaw_off=0.0)
    seated12 = bool(scene.tray_seated()[0])
    drop_bowl_at(front, (-0.040, 0.0))
    _REC["on"] = True
    drop_bowl_at(decoy, (0.062, 0.0))
    _report("contamination")
    _REC["on"] = False
    tgt_on = bool(scene._gather_tgt(scene.on_tray())[0])
    check("contamination: the TARGET correctly on the seated tray AND a decoy also "
          "on it — decoys_clear False kills success even though the loaded stage "
          "latched (score caps at 0.80)",
          seated12 and tgt_on and not bool(scene.decoys_clear()[0])
          and bool(scene._s_loaded[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= c.w_loaded + 1e-4)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.rack_tray_serve")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _nm, ok in checks)
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
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({exc!r})", flush=True)
        os._exit(1)

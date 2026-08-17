"""Smoke / rubric-REJECTION battery for GrooveLidChestScene — NullRobot, teleported
probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the physical claims the task rests on — the flanges bar entry
from above, the grooves hold the lid captive, the closure means nothing without the
contents — are load-bearing. Every probe is CONSTRUCTED (teleport, real physics steps,
judge) — instrumentation, never a solution: success() is monitored at EVERY step and
must never turn True anywhere in the battery (the audit is itself a check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; plate flat on the FLOOR (not
                           in the channel), cube inside; score 0, no success;
   2. randomization      — two seeded resets: READBACK chest xy+yaw, plate xy+yaw and
                           cube chest-frame xy all differ;
   3. side swap          — over 10 resets the plate spawns on BOTH sides of the chest;
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY      — the seed's own plan (bring the lid DOWN onto the box from
                           above, as with a hinged lid): the plate dropped flat
                           directly over the mouth lands ON the flanges (~22 mm above
                           the channel plane, readback) and can NEVER enter — nothing
                           latches, no success, score ~0;
   6. near-miss: partway — plate constructed IN the channel at mid-travel, settled:
                           `engaged`+`deep` latch (partial credit) but covered() is
                           False -> no success, score <= 0.75;
   7. porch seat         — plate gravity-dropped onto the porch only: `seated` latches
                           (score exactly 0.15), engagement depth still negative, no
                           success;
   8. MECHANISM: slide   — with the gold cube first REMOVED from the chest, the porch
                           drop + a real CoM push threads the plate home to the stop
                           (covered() True by contact dynamics) -> STILL no success:
                           the contents clause is load-bearing; score <= 0.75;
   9. MECHANISM: captive — from 8 (plate home): ~5.3 N straight UP (3x plate weight)
                           for 1.5 s lifts the plate < 12 mm — the flanges hold it
                           captive in the grooves (peak, measured during the hold);
  10. cube-on-lid        — correct closure but the cube parked ON the lid (~98 mm,
                           above the contents window) -> cube_in False, no success;
  11. settle gate        — the exact success pose but the plate still sliding at
                           ~0.35 m/s -> success refuses while anything moves (probe
                           dismantled before it can settle into a real success);
  12. rejection audit    — success() was never True at any step of this battery;
  13. score-cap audit    — score never exceeded 0.75 anywhere in this battery;
  14. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_box_i300.smoke --headless
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

_qapply, encode_force = task_scene._qapply, task_scene.encode_force

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
    p = scene.plate_local()[0]
    print(f"[smoke] {tag:18s} | plate_local=({float(p[0]):+.3f},{float(p[1]):+.3f},"
          f"{float(p[2]):+.3f}) depth={float(scene.lead_in()[0]):+.3f} "
          f"chan={bool(scene.in_channel()[0])} covered={bool(scene.covered()[0])} "
          f"cube_in={bool(scene.cube_in()[0])} "
          f"seat={bool(scene._seated[0])} eng={bool(scene._engaged[0])} "
          f"deep={bool(scene._deep[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.groove_lid_chest")().build(num_envs=args.num_envs,
                                                      device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.30, -0.65, 0.55)) + o),
                                tuple(np.array((0.45, 0.00, 0.06)) + o),
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

    def chest_pt(loc_xyz) -> torch.Tensor:
        """Chest-frame point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(list(loc_xyz), device=device).expand(n, 3)
        return scene.chest.data.root_pos_w + _qapply(scene.chest.data.root_quat_w, loc)

    def q_chest() -> torch.Tensor:
        _refresh()
        return scene.chest.data.root_quat_w.clone()

    def drop_on_porch() -> None:
        """Gravity-drop the aligned plate from a free hover over the porch."""
        _write_body(scene.plate, chest_pt((c.drop_x, 0.0, c.z_seat + 0.030)),
                    q_chest())
        _step(180)

    def px_now() -> float:
        _refresh()
        return float(scene.plate_local()[0, 0])

    def slide_home(tag: str) -> bool:
        """Real CoM push (velocity-regulated, force-frame probed) until the plate
        reaches the stop. Same actuation as solve.py — the mechanism, not a write."""
        qc = q_chest()
        u3 = _qapply(qc, torch.tensor([[-1.0, 0.0, 0.0]], device=device).expand(n, 3))
        u_xy = u3[0, :2] / max(float(u3[0, :2].norm()), 1e-6)
        q_ref = scene.plate.data.root_quat_w.clone()
        mode = 0
        floor_f = 0.8
        zero = torch.zeros(n, 1, 3, device=device)
        win_i, win_px = 0, px_now()
        for i in range(2400):
            px = px_now()
            if px <= c.px_home + 0.004:
                scene.plate.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
                return True
            v = scene.plate.data.root_lin_vel_w[0, :2]
            v_along = float((v * u_xy).sum())
            f_along = 6.0 * (0.08 - v_along)
            if abs(v_along) < 0.02 and f_along < floor_f:
                f_along = floor_f
            f_along = min(max(f_along, -1.0), 4.0)
            f_world = torch.zeros(n, 3, device=device)
            f_world[0, :2] = u_xy * f_along
            f_arg = encode_force(mode, q_ref, scene.plate.data.root_quat_w, f_world)
            scene.plate.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero, env_ids=_all_ids(), is_global=True)
            _step(1)
            if i - win_i >= 45:
                px2 = px_now()
                if px2 > win_px + 0.008:
                    mode = 1 - mode
                    print(f"[smoke] {tag}: moving away; force-frame mode -> {mode}",
                          flush=True)
                elif px2 > win_px - 0.004:
                    floor_f = min(floor_f + 0.4, 3.0)
                    print(f"[smoke] {tag}: stalled at px {px2:.3f}; floor -> "
                          f"{floor_f:.1f} N", flush=True)
                win_i, win_px = i, px2
        scene.plate.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        print(f"[smoke] {tag}: push timed out at px {px_now():.3f}", flush=True)
        return False

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; plate flat on the FLOOR (not in the "
          "channel), cube inside the cavity; score 0, no success",
          bool(scene._finite()[0]) and not bool(scene.in_channel()[0])
          and bool(scene.cube_in()[0]) and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.chest.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.chest.data.root_quat_w[0]),
                scene.plate.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.plate.data.root_quat_w[0]),
                scene._chest_local(scene.cube.data.root_pos_w)[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_cp, a_cy, a_pp, a_py, a_kp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_cp, b_cy, b_pp, b_py, b_kp = readback()
    d_cp, d_pp = float((a_cp - b_cp).norm()), float((a_pp - b_pp).norm())
    d_kp = float((a_kp - b_kp).norm())
    d_cy, d_py = dyaw(a_cy, b_cy), dyaw(a_py, b_py)
    print(f"[smoke] randomization deltas: chest_xy={d_cp * 1000:.1f}mm "
          f"chest_yaw={d_cy:.1f}deg plate_xy={d_pp * 1000:.1f}mm "
          f"plate_yaw={d_py:.1f}deg cube_local_xy={d_kp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: chest xy+yaw, plate xy+yaw and cube chest-frame xy "
          "readback differ across seeds",
          d_cp > 0.003 and d_cy > 2.0 and d_pp > 0.003 and d_py > 2.0 and d_kp > 0.002)

    # ================= 3. side swap ===============================================================
    sides = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sides.add("+y" if float(scene.plate_side[0]) > 0 else "-y")
    print(f"[smoke] over 10 resets: plate sides {sorted(sides)}", flush=True)
    check("side swap: the plate spawns on BOTH sides of the chest over 10 resets",
          sides == {"+y", "-y"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY (lid from above) ======================
    # The seed closes its box by bringing the hinged lid DOWN over the mouth. The
    # naive transfer here — lay the plate flat directly over the mouth and let it
    # drop — lands it ON the flanges, ~22 mm above the channel plane. The flanges
    # bar entry from above (the scene's central honesty assert, now demonstrated):
    # nothing latches, no credit, no success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_body(scene.plate, chest_pt((0.0, 0.0, c.flange_top + c.plate_t / 2 + 0.030)),
                q_chest())
    _step(180)
    _report("seed-strategy")
    p = scene.plate_local()[0]
    print(f"[smoke] top-dropped plate rests at local z={float(p[2]) * 1000:.1f}mm "
          f"(channel plane {c.z_seat * 1000:.0f}mm, flange top "
          f"{c.flange_top * 1000:.0f}mm)", flush=True)
    check("negative (SEED strategy): plate dropped flat over the mouth rests ON the "
          "flanges, ~22 mm above the channel — never enters: no latches, no success, "
          "score ~0",
          float(p[2]) > c.slot_top and not bool(scene.in_channel()[0])
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. near-miss: plate left partway down the channel ==========================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_body(scene.plate, chest_pt((0.060, 0.0, c.z_seat + 0.002)), q_chest())
    _step(90)
    _report("partway")
    px = float(scene.plate_local()[0, 0])
    print(f"[smoke] mid-travel plate px={px * 1000:.1f}mm "
          f"(covered needs <= {c.px_covered * 1000:.1f}mm)", flush=True)
    check("near-miss (partway): plate settled mid-channel — `engaged`+`deep` latch as "
          "partial credit but covered() is False: no success, score <= 0.75",
          bool(scene.in_channel()[0]) and px > c.px_covered
          and bool(scene._engaged[0]) and bool(scene._deep[0])
          and not bool(scene.covered()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.75 + 1e-4)

    # ================= 7. porch seat only =========================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    drop_on_porch()
    _report("porch-seat")
    depth = float(scene.lead_in()[0])
    s7 = float(scene.score()[0])
    print(f"[smoke] porch seat: depth={depth * 1000:.1f}mm (engaged needs "
          f">= {c.eng_min * 1000:.0f}mm), score={s7:.4f}", flush=True)
    check("porch seat: gravity-dropped plate rests on the porch rails — `seated` "
          "latches (score exactly 0.15), engagement still negative, no success",
          bool(scene._seated[0]) and depth < c.eng_min and not bool(scene._engaged[0])
          and abs(s7 - c.w_seat) <= 1e-3 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 8. MECHANISM: slide home with the cube removed =============================
    # Remove the gold cube from the chest FIRST (parked on open floor far away), then
    # execute the real closure: porch drop + CoM push to the stop. covered() turns
    # True by contact dynamics — and success still refuses. The check asserts the
    # push actually drove the plate home (a stalled probe fails, not vacuously).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    far = torch.zeros(n, 3, device=device)
    far[:, 0] = env.iscene.env_origins[:, 0] - 0.30
    far[:, 1] = env.iscene.env_origins[:, 1] + 0.50
    far[:, 2] = env.iscene.env_origins[:, 2] + c.cube / 2 + 0.003
    _write_body(scene.cube, far, None)
    _step(30)
    _REC["on"] = True
    drop_on_porch()
    ok8 = bool(scene.seated_now()[0]) and slide_home("mech-slide")
    _step(150)
    _report("mech-slide")
    check("MECHANISM (slide, contents removed): porch drop + real CoM push threads "
          "the plate home — covered() True by contact dynamics, but the cube is out: "
          "no success, score <= 0.75 (the contents clause is load-bearing)",
          ok8 and bool(scene.covered()[0]) and bool(scene.settled()[0])
          and not bool(scene.cube_in()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.75 + 1e-4)
    _REC["on"] = False

    # ================= 9. MECHANISM: the flanges hold the lid captive =============================
    # Continue from 8 (plate home in the grooves): pull straight UP with 3x the
    # plate's weight for 1.5 s. Peak rise is measured DURING the hold (gravity would
    # restore the pose before a post-release readback): the plate lifts a few mm to
    # the flange undersides and stops — captive.
    zero = torch.zeros(n, 1, 3, device=device)
    z_peak = 0.0
    f_up = torch.zeros(n, 1, 3, device=device)
    f_up[0, 0, 2] = 3.0 * c.plate_mass * 9.81
    for _ in range(180):
        scene.plate.set_external_force_and_torque(f_up, zero, env_ids=_all_ids(),
                                                  is_global=True)
        _step(1)
        z_peak = max(z_peak, float(scene.plate_local()[0, 2]))
    scene.plate.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
    _step(60)
    _report("captive-raid")
    print(f"[smoke] captive raid: peak plate local z={z_peak * 1000:.1f}mm "
          f"(channel plane {c.z_seat * 1000:.0f}mm, escape needs > "
          f"{(c.flange_top + c.plate_t / 2) * 1000:.0f}mm)", flush=True)
    check("MECHANISM (captive): 5.3 N straight up (3x plate weight) for 1.5 s lifts "
          "the home plate < 12 mm — the flanges hold it in the grooves",
          z_peak < c.z_seat + 0.012 and z_peak > 0.0
          and float(scene.plate_local()[0, 1].abs()) < c.wall_in
          and not bool(scene.success()[0]))

    # ================= 10. negative: cube parked ON the lid =======================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    # Write plate-home and cube-on-lid back to back with NO steps in between: a
    # stepped prefix with the cube still inside would BE a genuine success.
    _write_body(scene.plate, chest_pt((c.px_home + 0.002, 0.0, c.z_seat + 0.002)),
                q_chest())
    _write_body(scene.cube,
                chest_pt((0.0, 0.0, c.z_seat + c.plate_t / 2 + c.cube / 2 + 0.005)),
                None)
    _step(120)
    _report("cube-on-lid")
    kz = float(scene._chest_local(scene.cube.data.root_pos_w)[0, 2])
    print(f"[smoke] cube on the closed lid: local z={kz * 1000:.1f}mm (contents "
          f"window {c.contents_z_lo * 1000:.0f}..{c.contents_z_hi * 1000:.0f}mm)",
          flush=True)
    check("negative (cube on lid): correct closure but the cube rides ON the lid, "
          "above the contents window — cube_in False, no success",
          bool(scene.covered()[0]) and kz > c.contents_z_hi
          and not bool(scene.cube_in()[0]) and not bool(scene.success()[0]))

    # ================= 11. settle gate ============================================================
    # The exact success pose — plate home, cube inside — but the plate still SLIDING
    # out of the channel at ~0.35 m/s. success() must refuse while anything moves.
    # The probe is dismantled (transport) well before friction could stop it into a
    # real success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    qc = q_chest()
    u3 = _qapply(qc, torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))
    vel = torch.zeros(n, 3, device=device)
    vel[:, :2] = 0.35 * u3[:, :2]
    _write_body(scene.plate, chest_pt((c.px_home + 0.002, 0.0, c.z_seat + 0.002)),
                qc, lin_vel=vel)
    moving_ok = True
    for _ in range(3):
        _step(1)
        v = float(scene.plate.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v > 0.1 and not bool(scene.settled()[0]) \
            and not bool(scene.success()[0])
    _report("settle-gate")
    # dismantle before it can settle into a real success
    off = torch.zeros(n, 3, device=device)
    off[:, 0] = env.iscene.env_origins[:, 0] - 0.20
    off[:, 1] = env.iscene.env_origins[:, 1] - 0.50
    off[:, 2] = env.iscene.env_origins[:, 2] + c.plate_t / 2 + 0.003
    _write_body(scene.plate, off, None)
    _step(30)
    check("settle gate: the exact success pose still sliding at ~0.35 m/s is refused "
          "while anything moves",
          moving_ok)

    # ================= 12+13. audits ==============================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)
    print(f"[smoke] max score observed anywhere in the battery: {_AUD['smax']:.4f}",
          flush=True)
    check("score-cap audit: score never exceeded 0.75 anywhere in this battery",
          _AUD["smax"] <= 0.75 + 1e-4)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.groove_lid_chest")
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

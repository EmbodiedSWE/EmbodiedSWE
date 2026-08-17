"""Smoke / rubric-REJECTION battery for ShiftParkGrillScene — NullRobot, teleported
probes.

solve.py is the acceptance proof (slide back, raise, slide forward, gravity-park on
the rest bar, deliver the patty through the open mouth). This battery proves the
rubric REJECTS wrong outcomes and that the physical claims the task rests on — the
catch jams the spawn-position upswing, no over-center rest exists (a lid released at
the rear falls fully closed), the top mouth is the only patty aperture — are
load-bearing. Every probe is CONSTRUCTED (teleport, real physics steps, judge);
success() is monitored at EVERY step and must never turn True anywhere in the battery
(the audit is itself a check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite: lid forward-closed under the
                           catch, patty on the board; score 0, no success;
   2. randomization      — three seeded resets: READBACK grill xy+yaw, lid slide and
                           patty grill-frame xy; max-pairwise deltas real;
   3. null-policy        — 240 idle steps: score ~0, no success;
   4. SEED STRATEGY      — the seed's own plan (swing the closed lid up in place): a
                           real lifting force on the spawn lid JAMS on the catch
                           within a few degrees (moved-assert on the peak, sampled
                           DURING the press) and falls back closed on release — no
                           latch, no credit;
   5. rear release       — the lid constructed RAISED at the rear then released falls
                           fully CLOSED (pitch cap is under over-center, and parked
                           requires the forward slide): the forward-slide leg of the
                           park is load-bearing;
   6. closed + inside    — patty constructed ON THE GRATE with the lid still closed:
                           patty_on_grate True but no success and score ~0 — the
                           lid-parking clause is load-bearing (and this state is
                           unreachable: the closed-lid hover gap under-sizes the
                           patty);
   7. patty on the lid   — patty dropped onto the CLOSED lid rests on the plate top,
                           far above the on-grate z window: no credit (the mouth is
                           the only aperture);
   8. parked, wrong place— the lid parked on the bar (constructed, settles onto the
                           cleat) with the patty still on the board, then on the
                           ground: `parked` partial credit only, never success;
   9. settle gate        — a patty flying through the parked mouth airspace is
                           refused while it moves (probe dismantled mid-air);
  10. rejection audit    — success() was never True at any step of this battery;
  11. score-cap audit    — score never exceeded 0.70 anywhere in this battery;
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.open_grill_i343.smoke --headless
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

_qapply, _qmul, _qinv = task_scene._qapply, task_scene._qmul, task_scene._qinv
_qy, encode_force = task_scene._qy, task_scene.encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    p = scene.patty_local()[0]
    print(f"[smoke] {tag:18s} | slide={float(scene.slide()[0]) * 1000:6.1f}mm "
          f"pitch={math.degrees(float(scene.pitch()[0])):+6.1f}deg "
          f"patty=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
          f"on_grate={bool(scene.patty_on_grate()[0])} parked={bool(scene.parked()[0])} "
          f"rel={bool(scene._released[0])} rai={bool(scene._raised[0])} "
          f"par={bool(scene._parked[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shift_park_grill")().build(num_envs=args.num_envs,
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
        env.sim.set_camera_view(tuple(np.array((-0.50, -0.85, 0.75)) + o),
                                tuple(np.array((0.35, 0.00, 0.20)) + o),
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

    def q_grill() -> torch.Tensor:
        _refresh()
        return scene.grill.data.root_quat_w.clone()

    def grill_pt(loc_xyz) -> torch.Tensor:
        _refresh()
        loc = torch.tensor(list(loc_xyz), device=device).expand(n, 3)
        return scene.grill.data.root_pos_w + _qapply(q_grill(), loc)

    def write_lid(slide_m: float, pitch_deg: float) -> None:
        """Write the lid CONSISTENTLY on its worn mount: pin at `slide_m` along the
        slide, plate pitched `pitch_deg` up (0 = closed). Transport only."""
        qg = q_grill()
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = grill_pt((c.pin_x_rear + slide_m, 0.0, c.pin_z))
        st[:, 3:7] = _qmul(qg, _qy(torch.full((n,), -math.radians(pitch_deg),
                                              device=device)))
        scene.lid.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def write_patty(pos_w: torch.Tensor, lin_vel: torch.Tensor | None = None) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = q_grill()
        if lin_vel is not None:
            st[:, 7:10] = lin_vel
        scene.patty.write_root_state_to_sim(st, _all_ids())
        _refresh()

    zero = torch.zeros(n, 1, 3, device=device)

    def lid_lift_press(tag: str, hold: int = 240) -> tuple[float, float]:
        """The seed's own action: a real LIFTING force at the closed lid's CoM (well
        above its weight). At the spawn the tip is tucked under the catch, so the
        upswing jams within a few degrees. Returns (peak pitch deg, gain deg from
        start); peak sampled DURING the press (gravity restores it afterwards)."""
        up = torch.zeros(n, 3, device=device)
        up[:, 2] = 1.0
        a0 = math.degrees(float(scene.pitch()[0]))
        peak = a0
        mode = 0
        q_ref = scene.lid.data.root_quat_w.clone()
        f_lift = 3.0 * c.lid_mass * 9.81
        for i in range(hold):
            f_arg = encode_force(mode, q_ref, scene.lid.data.root_quat_w, up * f_lift)
            scene.lid.set_external_force_and_torque(
                f_arg.view(n, 1, 3), zero, env_ids=_all_ids(), is_global=True)
            _step(1)
            peak = max(peak, math.degrees(float(scene.pitch()[0])))
            if i == 59 and peak < a0 + 0.2:
                mode = 1 - mode
                print(f"[smoke] {tag}: lid not moving; force-frame mode -> {mode}",
                      flush=True)
        scene.lid.set_external_force_and_torque(zero, zero, env_ids=_all_ids())
        print(f"[smoke] {tag}: lid pressed up from {a0:.2f} to peak {peak:.2f} deg "
              f"(catch jam predicted ~{c.jam_deg:.1f}, raised latch at "
              f"{c.raised_deg:.0f})", flush=True)
        return peak, peak - a0

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _AUD["on"] = True
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    sl0 = float(scene.slide()[0])
    th0 = math.degrees(float(scene.pitch()[0]))
    p0 = scene.patty_local()[0]
    check("settle/no-NaN: layout settles finite; lid forward-closed under the catch, "
          "patty on the side board; score 0, no success",
          bool(scene._finite()[0]) and sl0 > c.slide_s - c.slide_jitter - 0.006
          and abs(th0) < 3.0 and c.board_y0 - 0.05 < float(p0[1]) < c.board_y1 + 0.05
          and float(p0[2]) > c.board_h - 0.02 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 2. randomization is real (3-seed max-pairwise readback) ===================
    def readback():
        _refresh()
        return (scene.grill.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.grill.data.root_quat_w[0]),
                float(scene.slide()[0]),
                scene.patty_local()[0, :2].clone())

    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(10)
        obs.append(readback())
    d_g = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    d_y = max(dyaw(a[1], b[1]) for a in obs for b in obs)
    d_s = max(abs(a[2] - b[2]) for a in obs for b in obs)
    d_p = max(float((a[3] - b[3]).norm()) for a in obs for b in obs)
    print(f"[smoke] randomization max-pairwise deltas: grill_xy={d_g * 1000:.1f}mm "
          f"grill_yaw={d_y:.1f}deg slide={d_s * 1000:.2f}mm patty_xy={d_p * 1000:.1f}mm",
          flush=True)
    check("randomization-is-real: grill xy+yaw, lid slide and patty grill-frame xy "
          "readback differ across seeds",
          d_g > 0.004 and d_y > 2.0 and d_s > 0.0005 and d_p > 0.003)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED'S OWN STRATEGY (swing the lid up in place) ==========
    # The seed opens its grill by rotating the lid up where it sits. Here the same
    # action — a real lifting force on the spawned (forward-closed) lid — jams the
    # tip on the catch overhang within a few degrees and gravity re-closes it on
    # release. The central honesty probe: it must genuinely move (moved-assert) and
    # genuinely fail.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    peak, gain = lid_lift_press("seed-strategy")
    _step(120)
    _report("seed-strategy")
    _REC["on"] = False
    back = math.degrees(float(scene.pitch()[0]))
    sl4 = float(scene.slide()[0])
    check("negative (SEED strategy): a real lifting press on the spawn lid moves it "
          "then JAMS on the catch a few degrees up, far short of the `raised` latch, "
          "and re-closes on release — no latch, no credit, no success",
          gain >= 0.3 and peak <= c.jam_deg + 4.0 and back < 3.0
          and sl4 >= c.slide_fwd_min
          and not bool(scene._raised[0]) and not bool(scene._released[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 5. rear release falls fully closed =========================================
    # RAISED at the REAR (the only place the sweep clears the rest bar) and released:
    # the pitch cap is under over-center, so gravity closes the lid completely. There
    # is no rest at the rear — the forward slide is load-bearing for the park.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_lid(0.002, 82.0)
    _step(420)
    _report("rear-release")
    _REC["on"] = False
    th5 = math.degrees(float(scene.pitch()[0]))
    check("negative (rear release): the lid raised at the rear and released falls "
          "fully CLOSED (no over-center rest; parked needs the forward slide) — "
          "`raised`+`released` latch at most, no park, no success",
          th5 < c.released_low_deg and not bool(scene.parked()[0])
          and not bool(scene._parked[0]) and float(scene.score()[0]) <= 0.36
          and not bool(scene.success()[0]))

    # ================= 6. negative: patty on the grate but the lid still CLOSED ===================
    # (Unreachable through any aperture — the closed-lid hover gap under-sizes the
    # patty — but constructible by teleport.) patty_on_grate holds, success must not:
    # the lid-parking clause is load-bearing.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_patty(grill_pt((0.0, 0.0, c.grate_z + c.patty_t / 2 + 0.003)))
    _step(120)
    _report("closed+inside")
    check("negative (closed lid, patty inside): patty constructed on the grate under "
          "the CLOSED lid — patty_on_grate True but the lid is not parked: no "
          "success, score ~0",
          bool(scene.patty_on_grate()[0]) and not bool(scene.parked()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 7. negative: patty dropped on the CLOSED lid ===============================
    # The mouth is the only aperture: a patty delivered onto the closed grill rests
    # on the plate top, far above the on-grate z window.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    write_patty(grill_pt((0.05, 0.0, 0.28)))
    _step(300)
    _report("patty-on-lid")
    p7 = scene.patty_local()[0]
    check("negative (patty on the closed lid): dropped onto the closed grill it "
          "rests ON the plate top, far above the on-grate z window — no credit",
          float(p7[2]) > c.patty_z_hi + 0.02 and abs(float(p7[0])) < 0.20
          and not bool(scene.patty_on_grate()[0]) and not bool(scene.success()[0]))

    # ================= 8. parked lid, patty in the wrong place ====================================
    # Construct the park honestly: the lid is set just above its bar-rest pose and
    # settles onto the bar/cleat (real contact). With the patty still on the BOARD,
    # then on the GROUND, `parked` partial credit latches but success never fires.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    write_lid(0.066, 63.0)
    ok8_park = False
    for _ in range(600):
        _step(1)
        if bool(scene.parked()[0]) and bool(scene.settled()[0]):
            ok8_park = True
            break
    _step(60)
    _report("parked+board")
    ok8a = ok8_park and bool(scene._parked[0]) \
        and abs(float(scene.score()[0]) - c.w_parked) <= 0.01 \
        and not bool(scene.success()[0])
    write_patty(grill_pt((0.30, -0.28, c.patty_t / 2 + 0.003)))
    _step(120)
    _report("parked+ground")
    _REC["on"] = False
    check("negative (parked, patty elsewhere): lid genuinely parked on the rest bar "
          "but the patty on the board / on the ground — `parked` partial credit "
          "only, never success",
          ok8a and not bool(scene.patty_on_grate()[0])
          and float(scene.score()[0]) <= c.w_parked + 0.01
          and not bool(scene.success()[0]))

    # ================= 9. settle gate =============================================================
    # Still parked from check 8: a patty FLYING through the mouth airspace (upward +
    # lateral velocity) must be refused while it moves. The probe is dismantled
    # mid-air (transport) before it could land into a real success.
    vel = torch.zeros(n, 3, device=device)
    vel[:, 2] = 0.8
    vel[:, :2] = 0.3 * _qapply(q_grill(), torch.tensor([[0.0, 1.0, 0.0]],
                                                       device=device).expand(n, 3))[:, :2]
    write_patty(grill_pt((0.08, 0.0, c.grate_z + c.patty_t / 2 + 0.010)), lin_vel=vel)
    moving_ok = True
    for _ in range(3):
        _step(1)
        v9 = float(scene.patty.data.root_lin_vel_w[0].norm())
        moving_ok = moving_ok and v9 > 0.1 and not bool(scene.settled()[0]) \
            and not bool(scene.success()[0])
    _report("settle-gate")
    write_patty(grill_pt((0.30, -0.28, c.patty_t / 2 + 0.003)))  # dismantle mid-air
    _step(60)
    check("settle gate: a patty moving through the parked mouth airspace is refused "
          "while anything moves (probe dismantled before it can land)",
          moving_ok)

    # ================= 10+11. audits ==============================================================
    check("rejection audit: success() was never True at any step of this battery",
          _AUD["hits"] == 0)
    print(f"[smoke] max score observed anywhere in the battery: {_AUD['smax']:.4f}",
          flush=True)
    check("score-cap audit: score never exceeded 0.70 anywhere in this battery",
          _AUD["smax"] <= 0.70 + 5e-4)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shift_park_grill")
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

    t = threading.Timer(10.0, lambda: os._exit(code))
    t.daemon = True
    t.start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

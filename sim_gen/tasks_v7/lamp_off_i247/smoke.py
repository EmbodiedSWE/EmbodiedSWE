"""Smoke / rubric-REJECTION battery for LampIsolatorScene — NullRobot, teleported probes.

solve.py is the acceptance proof (force-fetch the bar, blind force-insertion into the
recessed socket, velocity-servo lever throw released before the stop). This battery
proves the rubric REJECTS wrong outcomes and that the two strategic differentiators
from the seed are physically load-bearing: the LAMP is a decoy (the seed's entire
plan operates on it and earns nothing), and the recessed OVERCENTER hub can neither
be reached by a finger nor half-thrown (a push short of dead centre falls back; a
push past dead centre commits). Every probe is CONSTRUCTED as a settled state
(teleport, real physics steps, judge); constructed partial states may earn latched
partial credit but none may reach success() unless the switch genuinely passes the
dead centre.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; hub gravity-held on its ON
                           stop, bar lying cradled on the saddle; score ~0, no
                           success;
   2. randomization      — two seeded resets: READBACK panel yaw, panel xy, saddle
                           xy and lamp xy all differ;
   3. flags read back    — across seeds BOTH ball-grip directions and BOTH lamp
                           sides occur, and each stored flag matches a pose readback;
   4. null-policy        — 240 idle steps: hub holds ON, bar stays cradled, score
                           ~0, no success;
   5. SEED STRATEGY      — the seed's whole plan lives ON THE LAMP: a hard 25 N
                           shove displaces/topples the lamp — and earns nothing
                           (hub unmoved, score ~0, no success);
   6. finger exclusion   — a finger surrogate (palm + finger, real Franka-ish
                           proportions) driven 8 N straight down the bore axis:
                           the palm arrests on the plate rims with the fingertip
                           still short of the funnel mouth — no touch, no hub
                           motion, no credit. The recess forces the tool;
   7. slot near-miss     — the bar dropped tip-down into the slot BESIDE the hub:
                           in the slot is not in the socket — `inserted` never
                           latches, no success;
   8. dead-centre short  — hub+bar CONSTRUCTED at -4 deg (a push that stopped just
                           short of vertical), released: gravity returns it to the
                           ON stop — no success, only latched partial credit;
   9. overcenter commit  — hub+bar constructed at +25 deg (just past dead centre),
                           released hands-off: gravity alone completes the throw
                           onto the OFF stop and HOLDS it — genuine success. With
                           check 8 this is the bistability the task is built on;
  10. revocation         — from the thrown state, the hub written back to ON (bar
                           parked): success reads False again and the score falls
                           to the latched 0.70 cap — success is judged live;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.lamp_off_i247.smoke --headless
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

_qmul, _qy, _qz = task_scene._qmul, task_scene._qy, task_scene._qz

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
    phi = float(scene.hub_angle_deg()[0])
    tip = scene._hub_local(scene.bar.data.root_pos_w)[0]
    print(f"[smoke] {tag:16s} | phi={phi:+7.2f}deg "
          f"tip_hub=({float(tip[0]):+.3f},{float(tip[1]):+.3f},{float(tip[2]):+.3f}) "
          f"in_socket={bool(scene.bar_tip_in_socket()[0])} "
          f"fetched={bool(scene._fetched[0])} inserted={bool(scene._inserted[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.lamp_isolator")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.45, -1.05, 0.95)) + o),
                                tuple(np.array((0.05, 0.0, 0.24)) + o),
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

    def phi() -> float:
        _refresh()
        return float(scene.hub_angle_deg()[0])

    zero = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)

    def wrench(body, f_w: torch.Tensor, t_w: torch.Tensor) -> None:
        q = body.data.root_quat_w
        body.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).reshape(n, 1, 3),
            quat_apply_inverse(q, t_w).reshape(n, 1, 3))

    def bore_axis() -> torch.Tensor:
        _refresh()
        return quat_apply(scene.hub.data.root_quat_w, ez)

    def axle_world() -> torch.Tensor:
        _refresh()
        a = torch.tensor([0.0, 0.0, c.axle_z], device=device).expand(n, 3)
        return scene.panel.data.root_pos_w + quat_apply(scene.panel.data.root_quat_w, a)

    def set_hub_phi(deg: float, *, with_bar: bool) -> None:
        """Teleport the hub to angle `deg` (the hub origin sits ON the axle, so this
        is an axle-consistent pure pose write), optionally with the bar seated in
        the socket — the whole linkage is written together (teleporting one body of
        a jointed/nested pair gets depenetrated back by the other)."""
        q_h = _qmul(scene.panel.data.root_quat_w,
                    _qy(torch.full((n,), math.radians(deg), device=device)))
        _write_body(scene.hub, axle_world(), q_h)
        if with_bar:
            t = torch.tensor([0.0, 0.0, c.insert_tip_z], device=device).expand(n, 3)
            _write_body(scene.bar, scene.hub.data.root_pos_w + quat_apply(q_h, t), q_h)

    def park_bar() -> None:
        """Lay the bar flat on the ground far from everything."""
        p = torch.tensor([-1.5, -1.5, c.shaft_r + 0.002], device=device).expand(n, 3)
        p = p + env.iscene.env_origins
        _write_body(scene.bar, p, _qy(torch.full((n,), math.pi / 2, device=device)))

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    tip_z = float(scene.bar.data.root_pos_w[0, 2] - env.iscene.env_origins[0, 2])
    up_z = float(quat_apply(scene.bar.data.root_quat_w, ez)[0, 2])
    check("settle/no-NaN: layout settles finite; hub gravity-held on its ON stop, "
          "bar lying cradled on the saddle; score ~0, no success",
          bool(scene._finite()[0]) and abs(phi() - c.phi_on) < 2.0
          and 0.03 < tip_z < 0.10 and abs(up_z) < 0.30
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        pyaw = yaw_of(scene.panel.data.root_quat_w[0])
        pp = scene.panel.data.root_pos_w[0, :2].clone()
        sp = scene.saddle.data.root_pos_w[0, :2].clone()
        lp = scene.lamp.data.root_pos_w[0, :2].clone()
        return pyaw, pp, sp, lp

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_pp, a_sp, a_lp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_pp, b_sp, b_lp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_pp = float((a_pp - b_pp).norm())
    d_sp = float((a_sp - b_sp).norm())
    d_lp = float((a_lp - b_lp).norm())
    print(f"[smoke] randomization deltas: panel_yaw={d_yawv:.1f}deg "
          f"panel_xy={d_pp * 1000:.1f}mm saddle_xy={d_sp * 1000:.1f}mm "
          f"lamp_xy={d_lp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: panel yaw, panel xy, saddle xy and lamp xy "
          "readback all differ across seeds",
          d_yawv > 2.0 and d_pp > 0.003 and d_sp > 0.02 and d_lp > 0.03)

    # ================= 3. randomized flags occur and read back ====================================
    seen_ball = {True: 0, False: 0}
    seen_lamp = {True: 0, False: 0}
    consistent = True
    for sd in range(300, 310):
        torch.manual_seed(sd)
        env.reset()
        _step(5)
        bflag = bool(scene.ball_dir_p[0])
        lflag = bool(scene.lamp_side_p[0])
        ax_s = quat_apply_inverse(scene.saddle.data.root_quat_w,
                                  quat_apply(scene.bar.data.root_quat_w, ez))[0]
        lamp_p = scene._panel_local(scene.lamp.data.root_pos_w)[0]
        consistent &= (float(ax_s[0]) > 0) == bflag
        consistent &= (float(lamp_p[1]) > 0) == lflag
        seen_ball[bflag] += 1
        seen_lamp[lflag] += 1
        if all(seen_ball.values()) and all(seen_lamp.values()) and sd >= 303:
            break
    print(f"[smoke] flag counts over seeds: ball+x={seen_ball[True]} "
          f"ball-x={seen_ball[False]} lamp+={seen_lamp[True]} lamp-={seen_lamp[False]} "
          f"consistent={consistent}", flush=True)
    check("randomization (flags): both ball-grip directions and both lamp sides "
          "occur across seeds and each flag matches its pose readback",
          all(v > 0 for v in seen_ball.values())
          and all(v > 0 for v in seen_lamp.values()) and consistent)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    bar0 = scene.bar.data.root_pos_w[0].clone()
    _step(240)
    _report("null-policy")
    drift = float((scene.bar.data.root_pos_w[0] - bar0).norm())
    check("null-policy-fails: 240 idle steps, hub holds its ON stop, bar stays "
          "cradled (drift < 2 cm), score ~0, no success",
          abs(phi() - c.phi_on) < 2.0 and drift < 0.02
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY (act on the lamp) =====================
    # rlbench/lamp_off pokes the lamp's own switch. Here EVERYTHING done to the lamp
    # is worthless: shove it hard (25 N, real contact physics), displacing/toppling
    # it — the hub does not move and no credit appears.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    phi_before = phi()
    lamp0 = scene.lamp.data.root_pos_w[0].clone()
    _refresh()
    away = scene.lamp.data.root_pos_w - scene.panel.data.root_pos_w
    away[:, 2] = 0.0
    away = away / away.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    _REC["on"] = True
    no_action = torch.empty(0, device=device)
    for _ in range(120):
        _refresh()
        wrench(scene.lamp, 25.0 * away, torch.zeros(n, 3, device=device))
        _step(1)
    scene.lamp.set_external_force_and_torque(zero, zero)
    _step(120)
    _report("lamp-shoved")
    _REC["on"] = False
    lamp_moved = float((scene.lamp.data.root_pos_w[0, :2] - lamp0[:2]).norm())
    lamp_up = float(quat_apply(scene.lamp.data.root_quat_w, ez)[0, 2])
    print(f"[smoke] lamp shove: moved {lamp_moved * 1000:.0f}mm, up_z={lamp_up:+.2f}, "
          f"hub {phi_before:+.2f} -> {phi():+.2f} deg", flush=True)
    check("negative (SEED strategy): a 25 N shove on the lamp displaces/topples it "
          "— and earns NOTHING (hub unmoved on its stop, score ~0, no success)",
          (lamp_moved > 0.10 or lamp_up < 0.7) and abs(phi() - phi_before) < 1.5
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 6. finger exclusion: the recess forces the tool ============================
    # Drive the finger surrogate straight down the bore axis with a real 8 N press
    # (+ gravity compensation, orientation held): the 75 mm palm arrests on the
    # plate rims while the 55 mm finger is still short of the funnel mouth. The
    # probe MOVES (the press is not vacuous) but never touches the hub.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    phi_before = phi()
    b = bore_axis()
    start = axle_world() + b * (c.fun_z1 + 0.130)
    _write_body(scene.probe, start, scene.hub.data.root_quat_w)
    probe_m = float(scene.probe.root_physx_view.get_masses()[0].sum())
    p0 = scene.probe.data.root_pos_w[0].clone()
    _REC["on"] = True
    for _ in range(240):
        _refresh()
        b = bore_axis()
        v = scene.probe.data.root_lin_vel_w
        w = scene.probe.data.root_ang_vel_w
        axis = quat_apply(scene.probe.data.root_quat_w, ez)
        f_w = probe_m * 9.81 * ez - 8.0 * b - 6.0 * v
        t_w = 0.05 * torch.cross(axis, b, dim=-1) - 0.05 * w
        wrench(scene.probe, f_w, t_w)
        _step(1)
    scene.probe.set_external_force_and_torque(zero, zero)
    _step(30)
    _report("finger-probe")
    _REC["on"] = False
    moved = float((scene.probe.data.root_pos_w[0] - p0).norm())
    tip_hub = scene._hub_local(scene.probe.data.root_pos_w)[0]
    print(f"[smoke] finger probe: moved {moved * 1000:.0f}mm, tip at hub-local "
          f"({float(tip_hub[0]):+.3f},{float(tip_hub[1]):+.3f},{float(tip_hub[2]):+.3f}), "
          f"funnel top at z={c.fun_z1:.3f}, hub {phi_before:+.2f} -> {phi():+.2f} deg",
          flush=True)
    check("finger exclusion: an 8 N fingertip press down the bore axis arrests on "
          "the plate rims with the tip still short of the funnel mouth — no touch, "
          "no hub motion, no credit",
          moved > 0.06 and float(tip_hub[2]) > c.fun_z1 + 0.003
          and abs(phi() - phi_before) < 3.0
          and float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 7. near-miss: in the SLOT is not in the SOCKET =============================
    # Drop the bar tip-down into the open slot beside the hub: it falls to the
    # plinth and rattles in the slot — `inserted` must never latch.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    loc = torch.tensor([0.085, 0.0, 0.42], device=device).expand(n, 3)
    pos = scene.panel.data.root_pos_w + quat_apply(scene.panel.data.root_quat_w, loc)
    _write_body(scene.bar, pos, scene.panel.data.root_quat_w)
    _step(300)
    _report("slot-miss")
    _REC["on"] = False
    tip_pl = scene._panel_local(scene.bar.data.root_pos_w)[0]
    check("near-miss (slot): bar dropped tip-down into the slot BESIDE the hub "
          "falls and rattles there — inserted never latches, no success",
          bool(scene._finite()[0]) and float(tip_pl[2]) < 0.30
          and not bool(scene._inserted[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.26)

    # ================= 8. near-miss: a push SHORT of dead centre falls back =======================
    # Construct hub+bar at -4 deg (4 deg shy of vertical), zero velocity, hands
    # off: gravity returns the switch to its ON stop. Latched partial credit only.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    set_hub_phi(-4.0, with_bar=True)
    _step(300)
    _report("deadcentre-short")
    _REC["on"] = False
    s8 = float(scene.score()[0])
    check("near-miss (dead centre): hub+bar released at -4 deg falls BACK to the "
          "ON stop — no success, score only the latched partials (~0.43)",
          phi() <= -12.0 and not bool(scene.success()[0]) and 0.38 <= s8 <= 0.48)

    # ================= 9. overcenter commit: past dead centre gravity completes ===================
    # Construct hub+bar at +25 deg, zero velocity, hands off: gravity alone must
    # carry the switch onto its OFF stop and hold it — genuine, settled success.
    # (With check 8 this demonstrates the bistable overcenter physics.)
    _REC["on"] = True
    set_hub_phi(25.0, with_bar=True)
    _step(420)
    _report("overcenter")
    _REC["on"] = False
    commit_ok = phi() >= c.succ_min_deg and bool(scene.success()[0]) \
        and float(scene.score()[0]) >= 0.999
    check("overcenter commit: released at +25 deg (past dead centre) the switch "
          "completes hands-off onto its OFF stop and holds — genuine success",
          commit_ok)

    # ================= 10. revocation: success is judged live =====================================
    park_bar()
    set_hub_phi(c.phi_on, with_bar=False)
    _step(90)
    _report("revoked")
    s10 = float(scene.score()[0])
    check("revocation: hub written back to ON (bar parked) — success reads False "
          "and the score falls to the latched 0.70 cap",
          commit_ok and not bool(scene.success()[0]) and 0.6999 <= s10 <= 0.7000005)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.lamp_isolator")
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

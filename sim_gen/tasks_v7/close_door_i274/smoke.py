"""Smoke / rubric-REJECTION battery for SaggingGateScene — NullRobot probes.

solve.py is the acceptance proof (force-lift the gate up its worn-hinge slack,
torque-carry it shut held high, release it into the socket). This battery proves
the rubric REJECTS wrong outcomes and that the THRESHOLD — the task's strategic
differentiator from the seed — is physically load-bearing in both directions:
pushing without lifting is arrested by the apron wall, and the seated shoe
arrests a real reopening torque. Constructed partial states may earn latched
partial credit but none may reach success() unless the lift-carry-release is
genuinely performed (or its exact end state genuinely built, as in the
lock-reality check).

Checks:
   1. settle/no-NaN      — seeded reset settles finite; the gate hangs at its
                           random angle at the BOTTOM of the hinge play; score ~0,
                           no success;
   2. randomization      — two seeded resets: READBACK frame yaw, frame xy and
                           gate initial angle all differ;
   3. null-policy        — 240 idle steps: nothing moves, score ~0, no success;
   4. SEED STRATEGY      — the seed's whole plan ("push the door shut"): the
                           solve's own closing torque servo WITHOUT the lift —
                           the sagging shoe jams face-on against the apron outer
                           wall at ~52 deg, far outside the closed tolerance;
                           no success, score stays small;
   5. slam no-vault      — a 5 N.m closing slam (4x the servo clamp): the impact
                           must not convert swing momentum into enough heave to
                           vault the apron — the gate still ends open, never
                           success;
   6. proud near-miss    — gate CONSTRUCTED resting proud ON the apron top at
                           16 deg (the botched-drop outcome): settled, but the
                           shoe z readback is far above the seat window — not
                           seated, no success;
   7. early release      — gate released from carry height at 10 deg, shoe
                           straddling the socket's rim wall: it lands supported
                           by the rim, proud — not seated, no success;
   8. lift-only          — the solve's lift servo alone, then hands off: the
                           gate rides up and sags back, still open; `lifted`
                           latches but score <= 0.21, no success;
   9. held-high closed   — gate held CLOSED at carry height by the solve's own
                           servo (the closed-angle-only end state): the angle
                           clause holds but the shoe is not seated — no success
                           while held;
  10. lock-reality       — the true end state constructed (gate seated, shoe in
                           the well): success reads true, and a 2 N.m reopening
                           pull CANNOT swing the gate past the closed tolerance
                           — the socket, not a score artifact, keeps it closed;
                           success persists after the pull;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.close_door_i274.smoke --headless
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
    ang = float(scene.open_angle_deg()[0])
    hv = float(scene.heave()[0])
    p = scene.shoe_frame_local()[0]
    print(f"[smoke] {tag:16s} | angle={ang:+6.2f}deg heave={hv:+.4f} "
          f"shoe=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
          f"seated={bool(scene.shoe_seated()[0])} settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.sagging_gate")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.80, 0.75)) + o),
                                tuple(np.array((0.40, 0.00, 0.20)) + o),
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

    def ang() -> float:
        _refresh()
        return float(scene.open_angle_deg()[0])

    def heave() -> float:
        _refresh()
        return float(scene.heave()[0])

    zero = torch.zeros(n, 1, 3, device=device)
    gate_m = float(scene.gate.root_physx_view.get_masses()[0].sum())
    no_action = torch.empty(0, device=device)

    def hands_off() -> None:
        scene.gate.set_external_force_and_torque(zero, zero)

    def gate_wrench(fz: float, tau: float) -> None:
        """World-frame wrench on the gate root (converted to body frame)."""
        _refresh()
        q = scene.gate.data.root_quat_w
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 2] = fz
        t_w = torch.zeros(n, 3, device=device)
        t_w[:, 2] = tau
        scene.gate.set_external_force_and_torque(
            quat_apply_inverse(q, f_w).reshape(n, 1, 3),
            quat_apply_inverse(q, t_w).reshape(n, 1, 3))

    def drive(steps: int, *, h_tgt: float | None, a_tgt: float | None,
              kp_h: float = 100.0, kd_h: float = 22.0,
              kp_a: float = 1.6, kd_a: float = 0.5, t_clamp: float = 1.2) -> tuple[float, float]:
        """The solve's OWN combined servo (same structure and gains): heave PD +
        gravity feedforward toward h_tgt (None = no lift force at all — the
        seed's flat push) and yaw PD torque toward a_tgt deg (None = no swing).
        Returns (min angle seen, max heave seen). Leaves the last wrench applied."""
        a_min, h_max = 1e9, -1e9
        for _ in range(steps):
            vz = float(scene.gate.data.root_lin_vel_w[0, 2])
            wz = float(scene.gate.data.root_ang_vel_w[0, 2])
            fz = 0.0
            if h_tgt is not None:
                fz = gate_m * 9.81 + kp_h * (h_tgt - heave()) - kd_h * vz
                fz = max(-10.0, min(35.0, fz))
            tau = 0.0
            if a_tgt is not None:
                tau = kp_a * math.radians(ang() - a_tgt) - kd_a * wz
                tau = max(-t_clamp, min(t_clamp, tau))
            gate_wrench(fz, tau)
            _step(1)
            a_min = min(a_min, ang())
            h_max = max(h_max, heave())
        return a_min, h_max

    def hinge_world() -> torch.Tensor:
        _refresh()
        h = torch.tensor([0.0, c.hinge_y, 0.0], device=device).expand(n, 3)
        return scene.frame.data.root_pos_w + quat_apply(scene.frame.data.root_quat_w, h)

    def set_gate(deg: float, hv: float) -> None:
        """Teleport the gate to opening angle `deg` at heave `hv` — the gate
        origin sits ON the hinge axis, so this is a hinge-consistent pure pose
        write (zero velocity) within the joint's limit box."""
        _refresh()
        up = torch.zeros(n, 3, device=device)
        up[:, 2] = hv
        pos = hinge_world() + quat_apply(scene.frame.data.root_quat_w, up)
        q = _qmul(scene.frame.data.root_quat_w,
                  _qz(torch.full((n,), -math.radians(deg), device=device)))
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3:7] = q
        scene.gate.write_root_state_to_sim(st, _all_ids())
        _refresh()

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    a0 = float(scene.a0[0])
    _REC["on"] = True
    _step(180)
    _report("settle")
    _REC["on"] = False
    check("settle/no-NaN: layout settles finite; the gate hangs at its random angle "
          "at the BOTTOM of the worn hinge's play; score ~0, no success",
          bool(scene._finite()[0]) and abs(ang() - a0) < 5.0 and heave() < 0.010
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.frame.data.root_quat_w[0]),
                scene.frame.data.root_pos_w[0, :2].clone(),
                float(scene.a0[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_fp, a_a0 = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_fp, b_a0 = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_fp = float((a_fp - b_fp).norm())
    d_a0 = abs(a_a0 - b_a0)
    print(f"[smoke] randomization deltas: frame_yaw={d_yawv:.1f}deg "
          f"frame_xy={d_fp * 1000:.1f}mm gate_a0={d_a0:.1f}deg", flush=True)
    check("randomization-is-real: frame yaw, frame xy and gate initial angle "
          "readback all differ across seeds",
          d_yawv > 2.0 and d_fp > 0.003 and d_a0 > 2.0)

    # ================= 3. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, the gate keeps its angle at the "
          "bottom of the play, score ~0, no success",
          abs(ang() - a0) < 5.0 and heave() < 0.010
          and float(scene.score()[0]) <= 0.03 and not bool(scene.success()[0]))

    # ================= 4. negative: the SEED'S OWN STRATEGY =======================================
    # "Close the door" — push the hinged panel shut, nothing else. Run the
    # solve's own closing servo WITHOUT the lift: the sagging shoe must jam
    # face-on against the apron's outer wall at ~52 deg (geometry:
    # asin((apron_x1 + shoe_half)/shoe_y) ~ 51 deg), far outside the closed
    # tolerance. The servo saturates its clamp the whole way, so the arrest is
    # the wall, not a gain stall — and the probe verifiably moved the gate.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    a_min, h_max = drive(600, h_tgt=None, a_tgt=-2.0)
    hands_off()
    _step(90)
    _report("seed-push-arrest")
    _REC["on"] = False
    arrest_lo = math.degrees(math.asin((c.apron_x1 + c.shoe_half) / c.shoe_y))
    arrest_hi = math.degrees(math.asin((c.apron_x1 + c.shoe_half * math.sqrt(2.0))
                                       / c.shoe_y))
    print(f"[smoke] flat push arrested at {ang():.1f} deg (min {a_min:.1f}, geometry "
          f"predicts ~{arrest_lo:.1f}-{arrest_hi:.1f}); heave max {h_max * 1000:.1f} mm",
          flush=True)
    check("negative (SEED strategy): the solve's own closing servo WITHOUT the "
          "lift verifiably moves the gate yet arrests ON the apron wall (final "
          "angle inside the geometric arrest band, far outside the closed "
          "tolerance) — no success, score stays small",
          (a0 - ang()) > 8.0 and a_min > c.carry_max_angle + 5.0
          and arrest_lo - 2.0 <= ang() <= arrest_hi + 4.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.25)

    # ================= 5. negative: a hard SLAM cannot vault the apron ============================
    # 5 N.m constant closing torque (4x the solve's clamp): the shoe strikes the
    # apron wall at speed. The hinge has only yaw+heave freedom and the wall is
    # vertical, so the impact must not convert into enough heave to carry the
    # shoe over the 35 mm step. The gate must still end open — never success.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    a_min, h_max = 1e9, -1e9
    for i in range(360):
        gate_wrench(0.0, 5.0)
        _step(1)
        a_min = min(a_min, ang())
        h_max = max(h_max, heave())
    hands_off()
    _step(120)
    _report("slam-arrest")
    _REC["on"] = False
    print(f"[smoke] slam: min angle {a_min:.1f} deg, max heave {h_max * 1000:.1f} mm "
          f"(vault needs {(c.apron_top - c.shoe_z0) * 1000:.0f} mm)", flush=True)
    check("negative (slam no-vault): a 5 N.m closing slam bounces off the apron "
          "wall without vaulting it — the gate ends open past the carry gate, "
          "no success",
          a_min > 30.0 and ang() > 30.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.35)

    # ================= 6. near-miss: dropped PROUD on the apron top ===============================
    # The botched-drop outcome: gate at 16 deg with the shoe resting ON the apron
    # top (constructed 4 mm above rest, settled). The gate is still and 'closed-ish'
    # but the shoe z readback sits ~30 mm above the seat window — not seated.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    set_gate(16.0, c.apron_top - c.shoe_z0 + 0.004)
    _step(150)
    _report("proud-rest")
    p = scene.shoe_frame_local()[0]
    check("near-miss (proud): gate resting with the shoe proud ON the apron top "
          "— settled but the shoe is ~30 mm above the seat window, not seated, "
          "no success",
          bool(scene.settled()[0]) and float(p[2]) > c.seat_z_max + 0.010
          and not bool(scene.shoe_seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.7000005)

    # ================= 7. near-miss: EARLY release straddles the rim ==============================
    # Released from carry height at 10 deg — the shoe's footprint straddles the
    # socket's rim wall. With only yaw+heave freedom it cannot tip in: it lands
    # supported by the rim, proud. Not seated, no success.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    set_gate(10.0, c.slack - 0.002)
    _step(180)
    _report("early-release")
    _REC["on"] = False
    p = scene.shoe_frame_local()[0]
    check("near-miss (early release): dropped at 10 deg the shoe lands ON the "
          "socket's rim wall and rests proud — not seated, no success",
          bool(scene.settled()[0]) and heave() > 0.020
          and float(p[2]) > c.seat_z_max + 0.010
          and not bool(scene.shoe_seated()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.7000005)

    # ================= 8. negative: LIFT ALONE is not the task ====================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    drive(240, h_tgt=c.slack - 0.002, a_tgt=None)
    lifted_mid = heave() > c.lift_heave
    hands_off()
    _step(150)
    _report("lift-only")
    check("negative (lift-only): the solve's lift servo alone raises the gate up "
          "the slack (lifted latches) but it sags back still open — score <= 0.21, "
          "no success",
          lifted_mid and bool(scene._lifted[0]) and heave() < 0.010
          and abs(ang() - a0) < 8.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.21)

    # ================= 9. near-miss: closed-but-HELD-HIGH is not success ==========================
    # The angle clause alone must not be the goal: hold the gate closed at carry
    # height with the solve's own servo — angle within tolerance, shoe hovering
    # 45+ mm above its seat — no success while held.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    set_gate(2.0, c.slack - 0.002)
    ok_held = True
    for _ in range(120):
        drive(1, h_tgt=c.slack - 0.002, a_tgt=1.9)
        ok_held = ok_held and not bool(scene.success()[0])
    held_ang, held_hv = ang(), heave()
    not_seated_held = not bool(scene.shoe_seated()[0])
    _report("held-high")
    hands_off()
    _step(150)  # release: it may genuinely seat now — that is the task's goal,
    _report("held-release")  # the check judged the HELD state above
    check("near-miss (held high): gate held CLOSED at carry height for 1 s — the "
          "angle clause holds but the shoe is not seated; never success while held",
          ok_held and held_ang <= c.closed_max_deg and held_hv > 0.035
          and not_seated_held)

    # ================= 10. lock-reality: the socket physically keeps the gate closed ==============
    # Construct the true end state (gate seated, shoe in the well; this exact
    # state is what the solve produced dynamically), verify success reads true,
    # then pull the gate open with 2 N.m: the shoe presses the well wall and the
    # gate must stay within the closed tolerance; success persists after the pull.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    seat_deg = math.degrees(math.asin(c.well_cx / c.shoe_y))
    set_gate(seat_deg, 0.001)
    _step(150)
    _report("seated-built")
    seated_built = bool(scene.shoe_seated()[0]) and bool(scene.success()[0])
    max_a = 0.0
    for i in range(300):
        gate_wrench(0.0, -2.0)  # opening pull (open = -z)
        env.step(no_action, render=False)
        if i % 10 == 0:
            max_a = max(max_a, ang())
    max_a = max(max_a, ang())
    hands_off()
    _step(120)
    _report("seat-held")
    _REC["on"] = False
    print(f"[smoke] seated gate under 2 N.m reopening pull: max angle {max_a:.2f} deg "
          f"(tolerance {c.closed_max_deg:.1f})", flush=True)
    check("lock-reality: the seated shoe arrests a 2 N.m reopening pull within "
          "the closed tolerance and success persists — the socket is load-bearing",
          seated_built and max_a <= c.closed_max_deg + 0.3
          and bool(scene.shoe_seated()[0]) and bool(scene.success()[0]))

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sagging_gate")
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

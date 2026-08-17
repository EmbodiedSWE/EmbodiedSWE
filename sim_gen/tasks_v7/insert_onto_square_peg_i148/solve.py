"""Teleport solution for HookEscapeScene (sim_gen task `insert_onto_square_peg_i148`) —
the task's legitimacy certificate.

Teleportation handles TRANSPORT ONLY. Every load-bearing interaction — the whole
constrained escape off the hook rail — happens through contact dynamics under applied
wrenches (a floating force/torque servo standing in for a gripper carry):

1. READBACK: settle, read the rack pose (the arm direction is randomized), the dish
   position, and the stacking order of the two rings.
2. (orange-on-top episodes) SHEPHERD THE DECOY: the same wrench servo walks the ORANGE
   ring up the post, pitches it ~90 deg through the elbow, runs it out along the arm and
   off the open tip; once free of the rail it is teleported (transport of a now-free
   object) to a park spot on the floor on the far side of the rack from the dish.
3. CLIMB+ELBOW: the BLUE ring is servo-walked up the post and pitched through the elbow
   turn — bore axis rotating from world-up to along-arm — the topological crux.
4. ARM RUN: the blue ring is servo-run along the horizontal arm, off the OPEN tip, and
   held hovering past it; `rail_dist` readback must certify it is FREE of the rail.
5. PLACE: the free blue ring is teleported flat a few cm above the dish floor
   (exactly what a carry delivers), released, and drops in under gravity.

Servo plant: F = Kp*(carrot - p) - Kd*v + m*g*z_hat (carrot advances toward the active
waypoint at ~0.07 m/s, holding on error), tau = Kq*(axis x target) - Kw*omega. Gains
respect the one-substep wrench delay (Kd*dt/m ~ 0.31, Kw*dt/I ~ 0.4). Pod force-frame
quirk: applied wrenches are dragged by the body's rotation-since-reset, so both F and
tau are pre-encoded with R_ref * R_now^T (mode 1, default) — guarded by a runtime
progress probe that flips the mode, then escalates GAINS (not just caps) on stall.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: stage credit is
latched in the scene), then holds HANDS-OFF >= 3.3 simulated seconds after success()
first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still holds.

Run (forge): python -u -m simgen_tasks.insert_onto_square_peg_i148.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import math
import os
import threading

import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod

_qapply = scene_mod._qapply
_qmul = scene_mod._qmul
_qinv = scene_mod._qinv

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

_DT = 1.0 / 120.0
_M = scene_mod._RING_M  # 0.08 kg
_G = 9.81
# force PD (Kd*dt/m ~ 0.31 << 1: stable under the one-substep wrench delay)
_KP, _KD, _FMAX = 50.0, 3.0, 2.5
# axis-alignment torque PD (Kw*dt/I ~ 0.4 with I_transverse ~ 8.4e-5)
_KQ, _KW, _TMAX = 0.04, 0.004, 0.08
_V_CARROT = 0.07  # m/s waypoint pacing
_Z_ARM = scene_mod._Z_ARM
_ARM_TIP = scene_mod._ARM_TIP


def _wp(x, z, pitch_deg):
    """(center_rack, bore-axis_rack) waypoint: axis pitched about rack +y from +z."""
    a = math.radians(pitch_deg)
    return ((x, 0.0, z), (math.sin(a), 0.0, math.cos(a)))


# Escape path in the rack frame (verified against the rail geometry in scene.py: at
# every waypoint the rail crosses the ring plane inside the bore, and the pitched
# washer clears the post because the post ENDS flush with the arm top).
_WPS_CLIMB = [  # up the post, then the 90-deg elbow turn
    _wp(0.000, 0.150, 0.0),
    _wp(0.000, 0.230, 0.0),
    _wp(0.000, 0.300, 0.0),
    _wp(0.012, 0.322, 30.0),
    _wp(0.020, 0.330, 45.0),
    _wp(0.032, _Z_ARM, 65.0),
    _wp(0.048, _Z_ARM, 90.0),
]
_WPS_ARM = [  # out along the arm, off the OPEN tip, hover well past it
    _wp(0.085, _Z_ARM, 90.0),
    _wp(0.120, _Z_ARM, 90.0),
    _wp(0.155, _Z_ARM, 90.0),
    _wp(0.210, _Z_ARM, 90.0),
    _wp(0.300, _Z_ARM, 90.0),
]


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.hook_escape")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap).
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_wrench(body) -> None:
        body.set_external_force_and_torque(zero_wrench, zero_wrench, env_ids=all_ids)

    def report(tag: str) -> None:
        db = float(scene.rail_dist(scene.ring_blue)[0])
        do = float(scene.rail_dist(scene.ring_orange)[0])
        print(f"[solve] {tag:12s} | rail_d(blue)={db:.3f} rail_d(orange)={do:.3f} "
              f"in_dish={bool(scene.in_dish(scene.ring_blue)[0])} "
              f"decoy_clear={bool(scene.decoy_clear()[0])} "
              f"latch(c={float(scene.l_climb[0]):.2f},a={float(scene.l_arm[0]):.2f},"
              f"f={float(scene.l_free[0]):.2f},d={float(scene.l_dish[0]):.2f}) | "
              f"success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    # --- floating wrench servo ----------------------------------------------------------
    mode = [1]   # 1: pre-encode R_ref*R_now^T (pod drag quirk, default); 0: raw world
    scale = [1.0]  # gain escalation on stall (escalate GAIN, not just caps)
    q_ref = {}   # body -> reference quat for the drag pre-encode

    def apply_wrench(body, f_world: torch.Tensor, t_world: torch.Tensor) -> None:
        f, t = f_world.clone(), t_world.clone()
        if mode[0] == 1:
            dq = _qmul(q_ref[body], _qinv(body.data.root_quat_w))
            f = _qapply(dq, f)
            t = _qapply(dq, t)
        body.set_external_force_and_torque(f.view(n, 1, 3), t.view(n, 1, 3),
                                           env_ids=all_ids, is_global=True)

    def servo_to(body, wp, tag: str, max_steps: int = 1200) -> bool:
        """Drive `body` to a (center, axis) rack-frame waypoint with the carrot servo.
        Returns True once inside pos/axis tolerance; probes for wrong wrench frame
        (flip mode) and stalls (escalate gains)."""
        rp, rq = scene.rack.data.root_pos_w, scene.rack.data.root_quat_w
        tgt = rp + _qapply(rq, torch.tensor(wp[0], device=device).expand(n, 3))
        axis = _qapply(rq, torch.tensor(wp[1], device=device).expand(n, 3))
        carrot = body.data.root_pos_w.clone()
        anchor_d, anchor_i, flipped = None, 0, False
        for i in range(max_steps):
            p = body.data.root_pos_w
            v = body.data.root_lin_vel_w
            w = body.data.root_ang_vel_w
            q = body.data.root_quat_w
            a = _qapply(q, ez.expand(n, 3))
            sgn = torch.sign((a * axis).sum(-1, keepdim=True))
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            t_eff = axis * sgn
            d_tgt = float((p - tgt)[0].norm())
            ax_ok = float((a[0] * t_eff[0]).sum()) >= math.cos(math.radians(15.0))
            if d_tgt < 0.020 and ax_ok:
                return True
            # carrot: advance toward the waypoint at _V_CARROT, hold on error
            err = (carrot - p).norm(dim=-1, keepdim=True)
            delta = tgt - carrot
            dn = delta.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            adv = torch.where(err < 0.030, dn.clamp(max=_V_CARROT * _DT), dn * 0.0)
            carrot = carrot + delta / dn * adv
            # force PD + gravity feedforward
            f = _KP * scale[0] * (carrot - p) - _KD * v
            fn = f.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f = f * (fn.clamp(max=_FMAX * scale[0]) / fn)
            f = f + _M * _G * ez
            # axis-alignment torque PD
            t = _KQ * scale[0] * torch.cross(a, t_eff, dim=-1) - _KW * w
            tn = t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            t = t * (tn.clamp(max=_TMAX * scale[0]) / tn)
            apply_wrench(body, f, t)
            env.step(no_action)
            # progress probe every 1.5 s: wrong frame first, then gain stall
            if i - anchor_i >= 180:
                if anchor_d is not None and d_tgt > anchor_d - 0.005:
                    if not flipped:
                        mode[0] = 1 - mode[0]
                        flipped = True
                        print(f"[solve] {tag}: no progress (d {anchor_d:.3f}->{d_tgt:.3f}); "
                              f"wrench-frame mode -> {mode[0]}", flush=True)
                    else:
                        scale[0] = min(scale[0] * 1.5, 4.0)
                        print(f"[solve] {tag}: stalled at d={d_tgt:.3f}; "
                              f"gain scale -> {scale[0]:.2f}", flush=True)
                anchor_d, anchor_i = d_tgt, i
        print(f"[solve] {tag}: FAILED to reach wp {wp[0]} "
              f"(d={float((body.data.root_pos_w - tgt)[0].norm()):.3f})", flush=True)
        return False

    def escape(body, name: str, wps) -> None:
        for j, wp in enumerate(wps):
            assert servo_to(body, wp, f"{name}-wp{j}"), \
                f"{name}: escape stalled at waypoint {j} {wp[0]}"

    def teleport(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        """TRANSPORT ONLY: place a body at a pose with zero velocity."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    # ---------------- phase 0: settle + layout readback ---------------------------------
    step(240)
    rp = (scene.rack.data.root_pos_w - scene.env_origins)[0]
    q = scene.rack.data.root_quat_w[0]
    yaw = math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))
    dp = (scene.dish.data.root_pos_w - scene.env_origins)[0]
    blue_top = bool(scene.blue_top[0])
    zb = float(scene.ring_blue.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    zo = float(scene.ring_orange.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    assert (zb > zo) == blue_top, "stack-order readback disagrees with blue_top flag"
    print(f"[solve] layout readback (seed {args.seed}): rack=({float(rp[0]):+.3f},"
          f"{float(rp[1]):+.3f}) yaw={yaw:+.1f} deg | dish=({float(dp[0]):+.3f},"
          f"{float(dp[1]):+.3f}) | stack: {'BLUE on top' if blue_top else 'ORANGE on top'} "
          f"(z_blue={zb:.3f}, z_orange={zo:.3f})", flush=True)
    report("reset")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s_prev = print_score("P0 reset+settle")
    assert s_prev <= 0.03, f"baseline score should be ~0, got {s_prev}"
    q_ref[scene.ring_blue] = scene.ring_blue.data.root_quat_w.clone()
    q_ref[scene.ring_orange] = scene.ring_orange.data.root_quat_w.clone()

    # ---------------- phase 1: shepherd the orange decoy off first (if on top) ----------
    if not blue_top:
        escape(scene.ring_orange, "orange", _WPS_CLIMB + _WPS_ARM)
        assert bool(scene.free_of_rail(scene.ring_orange)[0]), \
            "orange finished the path but rail_dist says it is not free"
        clear_wrench(scene.ring_orange)
        # park the freed decoy on the floor, on the far side of the rack from the dish
        away = scene.rack.data.root_pos_w[:, :2] - scene.dish.data.root_pos_w[:, :2]
        away = away / away.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        park = scene.rack.data.root_pos_w.clone()
        park[:, :2] += away * 0.40
        park[:, 2] = scene.env_origins[:, 2] + scene_mod._RING_T / 2 + 0.03
        qid = torch.zeros(n, 4, device=device)
        qid[:, 0] = 1.0
        teleport(scene.ring_orange, park, qid)
        step(180)  # drop 3 cm, settle flat on the floor
        assert bool(scene.decoy_clear()[0]), "parked decoy violates the exclusion zone"
        report("P1-decoy")
        s_now = print_score("P1 orange decoy shepherded off the rail and parked")
        assert s_now >= s_prev - 1e-6, "score decreased across the decoy phase"
        s_prev = s_now
    else:
        print("[solve] blue is on top — the decoy stays put on the rack", flush=True)

    # ---------------- phase 2: climb the post + turn the elbow (blue) -------------------
    escape(scene.ring_blue, "blue", _WPS_CLIMB)
    report("P2-elbow")
    s_now = print_score("P2 blue ring climbed the post and turned the elbow")
    assert s_now >= s_prev - 1e-6, "score decreased across the climb"
    assert s_now >= 0.20, f"elbow turned but climb credit missing (score {s_now})"
    s_prev = s_now

    # ---------------- phase 3: run the arm, off the open tip ----------------------------
    escape(scene.ring_blue, "blue", _WPS_ARM)
    d_free = float(scene.rail_dist(scene.ring_blue)[0])
    assert bool(scene.free_of_rail(scene.ring_blue)[0]), \
        f"blue finished the path but rail_dist={d_free:.3f} says it is not free"
    report("P3-free")
    s_now = print_score("P3 blue ring ran off the open tip — free of the rail")
    assert s_now >= s_prev - 1e-6, "score decreased across the arm run"
    assert s_now >= 0.55, f"freed but arm/free credit missing (score {s_now})"
    s_prev = s_now

    # ---------------- phase 4: carry to the dish, release, settle -----------------------
    clear_wrench(scene.ring_blue)
    drop = scene.dish.data.root_pos_w.clone()
    drop[:, 2] = scene.env_origins[:, 2] + scene_mod._DISH_FLOOR_TOP \
        + scene_mod._RING_T / 2 + 0.030
    qid = torch.zeros(n, 4, device=device)
    qid[:, 0] = 1.0
    teleport(scene.ring_blue, drop, qid)  # TRANSPORT of a free object: hover flat, release
    step(300)  # 2.5 s hands-off: drop 3 cm, settle flat on the dish floor
    report("P4-dish")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after the dish drop)", flush=True)
        os._exit(1)
    s_now = print_score("P4 blue ring settled flat in the dish")
    assert s_now >= s_prev - 1e-6, "score decreased across the dish drop"
    s_prev = s_now

    # ---------------- persistence (>= 3 simulated seconds, no intervention) -------------
    hold, flickers = True, 0
    for i in range(400):  # 3.33 s at 120 Hz
        step(1)
        if not bool(scene.success()[0]):
            hold = False
            flickers += 1
            if flickers <= 8:
                p = scene._dish_local(scene.ring_blue.data.root_pos_w)[0]
                lv = float(scene.ring_blue.data.root_lin_vel_w[0].norm())
                av = float(scene.ring_blue.data.root_ang_vel_w[0].norm())
                print(f"[solve] persist flicker @step {i}: dish-local="
                      f"({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
                      f"in_dish={bool(scene.in_dish(scene.ring_blue)[0])} "
                      f"decoy_clear={bool(scene.decoy_clear()[0])} "
                      f"vel(lin={lv:.4f},ang={av:.4f})", flush=True)
    if flickers:
        print(f"[solve] persistence flickers: {flickers}/400 steps", flush=True)
    report("persist")
    s4 = print_score("P-persist persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s4 >= s_prev - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    except BaseException:  # noqa: BLE001 - die NOW, not at the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(1)

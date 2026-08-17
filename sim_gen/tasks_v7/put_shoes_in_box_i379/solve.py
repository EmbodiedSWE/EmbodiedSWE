"""Teleport solution for CrateFlipPackScene (sim_gen task `put_shoes_in_box_i379`) —
the task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. INVERT THE CRATE (pure contact dynamics — no teleport at all): the crate starts
   MOUTH-DOWN over the shoes. It is rolled mouth-up by TWO QUARTER-TIPS about its
   long bottom edge, each driven by an external torque about the crate's horizontal
   long axis (a fingertip push high on a wall, expressed as the equivalent CoM
   torque <= 3 N*m ~ a 7 N push at the top edge). The torque ENCODE FRAME is probed
   from the measured angular velocity (pod-dependent wrench-frame quirk) and the
   drive is a bang-bang loop with an angular-rate gate; past the balance angle the
   torque is cut and replaced by a rate-damping torque so the crate falls through
   and LANDS on the next face under gravity (a real pivot-fall-land contact event,
   twice). Landing poses are physical outcomes, not writes. Tip 1 uncovers the
   shoes; tip 2 erects the crate onto its base.
2. PACK (transport + gravity): each shoe is teleported to a free-space hover just
   above the OPEN mouth of the now-upright crate (position/heading read live from
   the crate's randomized pose) and RELEASED: it falls ~130 mm onto the crate floor
   and settles fully inside. Retries with small jogs if a landing perches.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
partial credit is latched), then holds HANDS-OFF for >= 3.3 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if it still
holds.

Run (forge): python -u -m simgen_tasks.put_shoes_in_box_i379.solve --headless [--seed N]
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
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as scene_mod

_qmul, _qz = scene_mod._qmul, scene_mod._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.crate_flip_pack")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_wrench = torch.zeros(n, 1, 3, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)

    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def crate_local(body) -> torch.Tensor:
        return quat_apply_inverse(scene.crate.data.root_quat_w,
                                  body.data.root_pos_w - scene.crate.data.root_pos_w)

    def up_z() -> float:
        return float(quat_apply(scene.crate.data.root_quat_w, ez)[0, 2])

    def theta() -> float:
        """Crate up-axis angle from world-up: pi mouth-down, pi/2 on side, 0 upright."""
        return math.acos(max(-1.0, min(1.0, up_z())))

    def report(tag: str) -> None:
        s = scene._status()
        parts = []
        for nm in scene.SHOE_NAMES:
            p = crate_local(scene.shoes[nm])[0]
            parts.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f})")
        print(f"[solve] {tag:12s} | up_z={float(s['up_z'][0]):+.3f} "
              f"crate_z={float(s['crate_z'][0]):+.4f} theta={theta():.3f} "
              + " ".join(parts)
              + f" inside={s['inside'][0].tolist()} covered={s['covered'][0].tolist()} "
              f"receptacle={bool(s['receptacle'][0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    def print_score(tag: str) -> float:
        sc = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {sc:.4f}", flush=True)
        return sc

    def settle(body, tag: str, max_steps: int = 480, min_steps: int = 40) -> bool:
        for i in range(max_steps):
            env.step(no_action)
            lv = float(body.data.root_lin_vel_w.norm(dim=-1)[0])
            av = float(body.data.root_ang_vel_w.norm(dim=-1)[0])
            if i > min_steps and lv < 0.02 and av < 0.2:
                return True
        print(f"[solve] {tag}: settle timeout (lv={lv:.3f} av={av:.3f})", flush=True)
        return False

    # ---------------- crate tipping controller (contact dynamics) ---------------------------
    # Torque encode-frame probe state (pod-dependent wrench-frame quirk):
    #   mode 0: pass the raw world torque vector; mode 1: body-encode it first.
    tq_mode = {"m": 0}

    def set_torque(t_world: torch.Tensor) -> None:
        t = t_world if tq_mode["m"] == 0 else quat_apply_inverse(
            scene.crate.data.root_quat_w, t_world)
        scene.crate.set_external_force_and_torque(
            zero_wrench, t.reshape(n, 1, 3), env_ids=all_ids, is_global=True)

    def clear_torque() -> None:
        scene.crate.set_external_force_and_torque(zero_wrench, zero_wrench,
                                                  env_ids=all_ids)

    def tip_axis(u_prev: torch.Tensor | None) -> torch.Tensor:
        """Horizontal unit vector along the crate's long (local x) axis, sign-aligned
        with the previous axis so the rotation continues the same way, (N,3)."""
        u = quat_apply(scene.crate.data.root_quat_w, ex)
        u[:, 2] = 0.0
        nn = u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        u = u / nn
        if u_prev is not None and float((u[0] * u_prev[0]).sum()) < 0.0:
            u = -u
        return u

    def omega_u(u: torch.Tensor) -> float:
        return float((scene.crate.data.root_ang_vel_w[0] * u[0]).sum())

    def probe_torque_frame() -> None:
        """Identify the wrench encode frame with a YAW-SPIN pulse. At mouth-down the
        crate's body z is exactly -world z, so a +z world torque given in the wrong
        frame comes out as -z: the SIGN of the measured w_z is the frame verdict.
        Spin only fights friction (~0.5 N*m), never gravity, so the pulse always
        moves and the verdict is never vacuous."""
        for tq in (1.2, 2.0, 3.0):
            tq_mode["m"] = 0
            set_torque(tq * ez)
            step(10)
            clear_torque()
            wz = float(scene.crate.data.root_ang_vel_w[0, 2])
            print(f"[solve] torque spin probe tq={tq:.1f} w_z={wz:+.3f}", flush=True)
            settle(scene.crate, "probe-recover", max_steps=360)
            if wz > 0.05:
                return                       # world vector passed through correctly
            if wz < -0.05:
                tq_mode["m"] = 1             # backend body-interprets: pre-encode
                print("[solve] torque frame -> body-encoded (mode 1)", flush=True)
                return
        raise AssertionError("torque frame probe never moved the crate")

    def quarter_tip(gate: float, tau0: float, tag: str) -> bool:
        """One quarter-roll about the long bottom edge: bang-bang torque drive (rate
        gate 2.5 rad/s, escalate on stall) until theta passes `gate` (past balance),
        then cut drive and rate-damp the gravity fall onto the next face."""
        u = tip_axis(tip_state["u"])
        tip_state["u"] = u.clone()
        tau = tau0
        th_mark = theta()
        stall_clock = 0
        driven = False
        for i in range(1200):
            th = theta()
            if th <= gate:
                driven = True
                break
            wu = omega_u(u)
            if wu > 2.5:
                clear_torque()
            else:
                set_torque(tau * u)
            env.step(no_action)
            stall_clock += 1
            if stall_clock >= 90:
                if th > th_mark - 0.02:  # no progress in 0.75 s: escalate
                    tau = min(tau * 1.45, 3.0)
                    print(f"[solve] {tag}: stall at theta={th:.3f}, "
                          f"tau -> {tau:.2f} N*m", flush=True)
                th_mark = th
                stall_clock = 0
        clear_torque()
        if not driven:
            print(f"[solve] {tag}: drive timeout at theta={theta():.3f}", flush=True)
            return False
        # damped descent: let gravity carry it through; bleed rotational energy
        for i in range(900):
            wu = omega_u(u)
            td = max(-0.6, min(0.6, -0.35 * wu))
            set_torque(td * u)
            env.step(no_action)
            lv = float(scene.crate.data.root_lin_vel_w.norm(dim=-1)[0])
            av = float(scene.crate.data.root_ang_vel_w.norm(dim=-1)[0])
            if i > 90 and lv < 0.02 and av < 0.2:
                break
        clear_torque()
        step(30)
        print(f"[solve] {tag}: landed at theta={theta():.3f} up_z={up_z():+.3f} "
              f"tau_end={tau:.2f}", flush=True)
        return True

    tip_state: dict[str, torch.Tensor | None] = {"u": None}

    # ---------------- shoe drop (transport + gravity) ---------------------------------------
    def crate_pose(x_loc: float, y_loc: float, z_loc: float, yaw_rel: float) -> torch.Tensor:
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = x_loc, y_loc, z_loc
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.crate.data.root_pos_w + quat_apply(
            scene.crate.data.root_quat_w, loc)
        st[:, 3:7] = _qmul(scene.crate.data.root_quat_w,
                           _qz(torch.full((n,), yaw_rel, device=device)))
        return st

    def drop_shoe(name: str, x_loc: float) -> None:
        """TRANSPORT to a free-space hover just above the open mouth, then hands-off:
        the shoe falls onto the crate floor. Small jogs retry a perched landing."""
        body = scene.shoes[name]
        idx = scene.SHOE_NAMES.index(name)
        hover_z = c.in_h + c.sole_t / 2 + 0.008
        for attempt in range(6):
            jx = 0.006 * ((attempt + 1) // 2) * (1 if attempt % 2 else -1) if attempt else 0.0
            body.write_root_state_to_sim(crate_pose(x_loc + jx, 0.0, hover_z, 0.0),
                                         all_ids)
            settle(body, name)
            step(30)
            if bool(scene._status()["inside"][0, idx]):
                return
            p = crate_local(body)[0]
            print(f"[solve] {name} drop attempt {attempt} not inside "
                  f"(loc={float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}); "
                  f"retrying", flush=True)
        print(f"SIM_GEN_SOLVE: FAIL ({name} would not land inside)", flush=True)
        os._exit(1)

    # ---------------- phase 0: reset, settle, baseline --------------------------------------
    step(120)
    cp = (scene.crate.data.root_pos_w - scene.env_origins)[0]
    cq = scene.crate.data.root_quat_w[0]
    spawns = []
    for nm in scene.SHOE_NAMES:
        p = (scene.shoes[nm].data.root_pos_w - scene.env_origins)[0]
        spawns.append(f"{nm}=({float(p[0]):+.3f},{float(p[1]):+.3f})")
    print(f"[solve] layout readback (seed {args.seed}): "
          f"crate=({float(cp[0]):+.3f},{float(cp[1]):+.3f}) up_z={up_z():+.3f} "
          f"quat=({float(cq[0]):+.3f},{float(cq[1]):+.3f},{float(cq[2]):+.3f},"
          f"{float(cq[3]):+.3f}) " + " ".join(spawns), flush=True)
    report("reset")
    for body in (scene.crate, *scene.shoes.values()):
        assert torch.isfinite(body.data.root_pos_w).all(), "NaN/inf after settle"
    assert up_z() < -0.95, "crate must start mouth-down"
    s_stat = scene._status()
    assert bool(s_stat["covered"][0].all()), "shoes must start covered by the crate"
    s0 = print_score("P0 reset+settle (crate inverted over the shoes)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: first quarter-tip — uncover the shoes ------------------------
    assert theta() > 3.0, f"unexpected start theta {theta():.3f}"
    probe_torque_frame()
    if not quarter_tip(1.85, 1.0, "tip1 (mouth-down->side)"):
        print("SIM_GEN_SOLVE: FAIL (tip 1 stalled)", flush=True)
        os._exit(1)
    settle(scene.crate, "tip1-settle")
    report("tip1")
    assert bool(scene._uncovered[0]), "uncover latch did not set after tip 1"
    assert not bool(scene.success()[0]), "uncovered alone cannot be success"
    s1 = print_score("P1 crate rolled onto its side — shoes uncovered")
    assert s1 >= s0 - 1e-6 and s1 >= 0.14, f"P1 score {s1} (expect >= 0.15)"

    # ---------------- phase 2: second quarter-tip — crate lands on its base -----------------
    for k in range(5):
        settle(scene.crate, "pre-tip", max_steps=360)
        if bool(scene._status()["receptacle"][0]):
            break
        th = theta()
        if th > 2.6:
            ok = quarter_tip(1.85, 1.0, f"tip2.{k} (mouth-down->side)")
        elif th > 0.8:
            ok = quarter_tip(1.05, 0.55, f"tip2.{k} (side->base)")
        else:
            step(120)
            ok = True
        if not ok:
            break
    settle(scene.crate, "erect-final")
    report("tip2")
    if not bool(scene._status()["receptacle"][0]):
        report("FAIL-erect")
        print("SIM_GEN_SOLVE: FAIL (crate would not stand on its base)", flush=True)
        os._exit(1)
    assert bool(scene._erected[0]), "erected latch did not set"
    assert not bool(scene.success()[0]), "empty upright crate cannot be success"
    s2 = print_score("P2 crate erected — standing upright on its base")
    assert s2 >= s1 - 1e-6 and s2 >= 0.29, f"P2 score {s2} (expect >= 0.30)"

    # ---------------- phase 3: shoe A dropped in --------------------------------------------
    drop_shoe("shoe_a", -0.08)
    report("shoe_a")
    assert bool(scene._packed[0, 0]), "shoe_a packed latch did not set"
    assert not bool(scene.success()[0]), "one shoe cannot be success"
    s3 = print_score("P3 shoe A inside the upright crate")
    assert s3 >= s2 - 1e-6 and s3 >= 0.49, f"P3 score {s3} (expect >= 0.50)"

    # ---------------- phase 4: shoe B dropped in — success ----------------------------------
    drop_shoe("shoe_b", +0.08)
    step(60)
    report("shoe_b")
    assert bool(scene._packed[0, 1]), "shoe_b packed latch did not set"
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after both shoes)", flush=True)
        os._exit(1)
    s4 = print_score("P4 both shoes inside, crate upright — success")
    assert s4 >= s3 - 1e-6 and s4 >= 0.99, f"P4 score {s4} (expect 1.0)"

    # ---------------- phase 5: persistence (>= 3.3 simulated seconds, hands-off) ------------
    hold = True
    for _ in range(10):  # 10 x 40 = 400 substeps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.3 s")
    ok = hold and bool(scene.success()[0]) and s5 >= s4 - 1e-6
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — Kit keeps the process alive; die loudly
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SOLVE: FAIL (exception)", flush=True)
        os._exit(2)

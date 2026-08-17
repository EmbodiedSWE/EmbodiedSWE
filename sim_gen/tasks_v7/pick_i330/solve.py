"""Solution for ChockRampScene (sim_gen task `pick_i330`) — the task's legitimacy
certificate.

Teleports are TRANSPORT ONLY (setting a free body down at a pose an arm reaches by
an ordinary pick-and-place); every load-bearing interaction is contact dynamics:

  P1  SEAT THE CHOCK (gravity + a fingertip press): the chock is lowered over the
      STARRED pocket — tab in the open socket mouth, plate bottom ~4 mm above the
      face — and RELEASED. Gravity drops it flush and, on the slick face, slides
      it downhill until the tab keys against the pocket's downhill wall; a brief
      2 N fingertip press along the downhill slope axis then confirms the key is
      hard against the wall (the wrench of a finger nudging the chock). Seating
      itself is never teleported: the drop, the key-up and the flush rest are all
      real contact.
  P2  GRAVITY DELIVERY (no force at all): the red cube is laid on the face a few
      mm UPHILL of the band (outside it — asserted) and RELEASED. The slick face
      does the transport: the cube slides down and is arrested by the seated
      chock's plate, coming to rest inside the band. The cube is never pushed and
      never touched again.
  P3  settle to live success (braced in the band + a >= 12-step stillness streak).
  P4  HANDS-OFF persistence >= 3.3 simulated seconds, then `SIM_GEN_SOLVE: SUCCESS`
      (the tab-in-pocket is what holds the park — nothing is pinned).

No teleport creates the goal: the chock staging hovers ABOVE the face with the tab
in the open socket mouth (an arm's insertion approach; the pocket is open from
above), and the cube staging is asserted OUTSIDE the band, unsupported — from
there physics alone decides. The smoke battery separately proves that without the
seated chock the same release ends on the floor past the ramp foot.

Prints `SIM_GEN_SCORE <s>` at each phase boundary; latched credit makes the
sequence non-decreasing: 0.00 -> 0.375 (seated) -> 0.75 (parked) -> 1.00.

Run (forge): python -u -m simgen_tasks.pick_i330.solve --headless [--seed N]
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
    from .scene import _qapply, _qinv
except ImportError:  # pragma: no cover - forge fallback
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import _qapply, _qinv

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.chock_ramp")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the statement on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def clear_forces() -> None:
        scene.chock.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        scene.cube.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)

    def rq() -> torch.Tensor:
        return scene.ramp.data.root_quat_w

    def chock_local() -> torch.Tensor:
        return scene._local(scene.chock.data.root_pos_w)[0]

    def cube_local() -> torch.Tensor:
        return scene._local(scene.cube.data.root_pos_w)[0]

    def report(tag: str) -> None:
        kl, cl = chock_local(), cube_local()
        print(f"[solve] {tag:12s} | chock=({float(kl[0]):+.4f},{float(kl[1]):+.4f},"
              f"{float(kl[2]):+.4f}) cube=({float(cl[0]):+.4f},{float(cl[1]):+.4f},"
              f"{float(cl[2]):+.4f}) seated={bool(scene.chock_seated()[0])} "
              f"band={bool(scene.on_band()[0])} braced={bool(scene.braced()[0])} "
              f"l_seat={bool(scene._l_seated[0])} l_park={bool(scene._l_parked[0])} "
              f"streak={int(scene._streak[0])} success={bool(scene.success()[0])} "
              f"score={float(scene.score()[0]):.3f}", flush=True)

    prev_score = [0.0]

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        assert s >= prev_score[0] - 1e-6, f"score decreased {prev_score[0]} -> {s}"
        prev_score[0] = s
        return s

    def place_local(body, local_xyz) -> None:
        """TRANSPORT ONLY: set a free body down at a slope-frame pose, aligned to the
        slope frame, zero velocity — what an arm does by pick-and-place."""
        st = torch.zeros(n, 13, device=device)
        loc = torch.tensor([list(local_xyz)], device=device, dtype=torch.float)
        st[:, 0:3] = scene.ramp.data.root_pos_w + _qapply(rq(), loc.expand(n, 3))
        st[:, 3:7] = rq()
        body.write_root_state_to_sim(st, all_ids)

    def press(body, u_local, mag: float, steps: int) -> None:
        """Fingertip press: a small constant force whose WORLD direction is the slope
        frame direction `u_local`, encoded per-step into the live body frame (immune
        to the pod's is_global stale-reference drag)."""
        u = torch.tensor([list(u_local)], device=device, dtype=torch.float)
        for _ in range(steps):
            f_w = _qapply(rq(), u)[0] * mag
            f = torch.zeros(n, 1, 3, device=device)
            f[0, 0, :] = _qapply(_qinv(body.data.root_quat_w), f_w.expand(n, 3))[0]
            body.set_external_force_and_torque(f, zero_w, env_ids=all_ids)
            env.step(no_action)
        clear_forces()
        env.step(no_action)

    # ---------------- phase 0: reset, settle, layout readback --------------------------------
    step(240)
    k_star = int(scene.k_star[0])
    x_star = float(scene.pockets_t[k_star])
    b_star = float(scene.bands_t[k_star])
    rp = (scene.ramp.data.root_pos_w - scene.env_origins)[0]
    q0 = rq()[0]
    # yaw readback from q = qz(yaw) * qy(-slope): project the slope x-axis
    xw = _qapply(q0.unsqueeze(0), torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
    ryaw = math.degrees(math.atan2(float(xw[1]), float(xw[0])))
    kl0, cl0 = chock_local(), cube_local()
    ml = scene._local(scene.marker.data.root_pos_w)[0]
    print(f"[solve] layout readback (seed {args.seed}): "
          f"ramp=({float(rp[0]):+.3f},{float(rp[1]):+.3f},{float(rp[2]):+.3f}) "
          f"yaw={ryaw:+.1f}deg k*={k_star} pocket_x={x_star:+.3f} band={b_star:+.3f} "
          f"marker_x={float(ml[0]):+.3f} swap={float(scene.swap[0]):+.0f}", flush=True)
    report("reset")
    # honesty readbacks: masses really authored (custom spawner!), bodies on the floor
    m_chock = float(scene.chock.root_physx_view.get_masses().sum())
    m_cube = float(scene.cube.root_physx_view.get_masses().sum())
    print(f"[solve] mass readback: chock={m_chock:.3f} kg cube={m_cube:.3f} kg", flush=True)
    assert abs(m_chock - c.chock_mass) < 0.02, f"chock mass not authored: {m_chock}"
    assert abs(m_cube - c.cube_mass) < 0.02, f"cube mass not authored: {m_cube}"
    assert abs(float(ml[0]) - b_star) < 0.005, "marker must sit on the starred band"
    assert bool(scene._finite()[0]), "NaN/inf after settle"
    assert not bool(scene.chock_seated()[0]), "chock must start unseated (on the floor)"
    assert not bool(scene.on_band()[0]), "cube must start off the ramp"
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    s0 = print_score("P0 reset+settle (both bodies on the floor, face empty)")
    assert s0 <= 0.01, f"baseline score should be ~0, got {s0}"

    # ---------------- phase 1: lower the chock into the starred pocket, gravity seats it -----
    # Hover: tab in the open socket mouth, plate bottom ~4 mm above the face; slightly
    # uphill of centre so the vertical-drop drift lands near centre, then gravity keys
    # the tab downhill against the pocket wall. The socket is open from above — this
    # is the arm's ordinary insertion approach, not a barrier crossing.
    place_local(scene.chock, (x_star + 0.002, 0.0, 0.004))
    step(120)
    report("chock-drop")
    # fingertip press downhill: key the tab hard against the pocket's downhill wall
    press(scene.chock, (-1.0, 0.0, -0.3), mag=2.0, steps=60)
    step(60)
    report("chock-press")
    kl = chock_local()
    if not bool(scene.chock_seated()[0]):
        print(f"SIM_GEN_SOLVE: FAIL (chock did not seat: {kl.tolist()})", flush=True)
        os._exit(1)
    assert bool(scene._l_seated[0]), "seated latch must be set"
    s1 = print_score("P1 chock keyed into the starred pocket (seated)")
    assert s1 >= c.w_seated - 1e-6, f"seated credit missing: {s1}"

    # ---------------- phase 2: lay the cube uphill of the band, gravity delivers it ----------
    x_rel = b_star + c.band_half + 0.006  # OUTSIDE the band's uphill edge
    assert x_rel > b_star + c.band_half + 0.002, "release must start outside the band"
    place_local(scene.cube, (x_rel, 0.0, c.cube_s / 2 + 0.004))
    assert not bool(scene.on_band()[0]), "release pose must not already be in the band"
    parked = False
    for i in range(600):  # the slide takes ~0.2 s; allow ample settle
        env.step(no_action)
        if bool(scene._l_parked[0]) and \
                float(scene.cube.data.root_lin_vel_w[0].norm()) < 0.05 and i > 30:
            parked = True
            break
    report("cube-slide")
    if not parked or not bool(scene.braced()[0]):
        print("SIM_GEN_SOLVE: FAIL (cube did not park against the chock)", flush=True)
        os._exit(1)
    s2 = print_score("P2 cube slid down and parked against the chock, inside the band")
    assert s2 >= 0.75 - 1e-6, f"parked credit missing: {s2}"

    # ---------------- phase 3: settle to live success ----------------------------------------
    step(240)
    report("settled")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (final state is not success)", flush=True)
        os._exit(1)
    s3 = print_score("P3 cube braced in the band, still (success)")
    assert s3 >= 1.0 - 1e-6, f"success must score 1.0, got {s3}"

    # ---------------- phase 4: persistence (>= 3.3 simulated seconds, hands-off) -------------
    hold = True
    for _ in range(10):  # 10 x 40 steps = 400 steps = 3.33 s at 120 Hz
        step(40)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s hands-off")
    ok = hold and bool(scene.success()[0]) and s4 >= s3 - 1e-6
    if ok:
        print("SIM_GEN_SOLVE: SUCCESS", flush=True)
    else:
        print("SIM_GEN_SOLVE: FAIL (success did not persist)", flush=True)

    code = 0 if ok else 1
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

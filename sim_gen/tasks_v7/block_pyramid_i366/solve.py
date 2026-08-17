"""Teleport solution for JengaQuarryScene (sim_gen task `block_pyramid_i366`) — the
task's legitimacy certificate.

The task's load-bearing interaction is the JENGA SLIDE: each red target block is
buried in the tower (bottom / middle layer), roofed by the perpendicular layer
above, and must be slid out lengthwise WITH THE SUPERSTRUCTURE'S WEIGHT RIDING ON
IT — while every cream block stays in its slot. How this solution executes it:

1. EXTRACTION (contact dynamics, never teleported): a velocity-capped horizontal
   force along the red block's own axis (the fingertip pull on its protruding end,
   <= a few N) slides it through its channel under the load of the layers above,
   until it emerges past the tower span and drops off the plinth onto the table.
   A weak lateral centring force keeps it from yawing into a corner jam; a stall
   escalates the force cap. The extraction direction for the bottom red points
   AWAY from the tray so nothing ever falls near it. THE TOWER IS NEVER TOUCHED:
   every cream block stays where physics leaves it, and the scene's
   tower_standing() readback is asserted after each phase.
2. TRANSPORT (teleport): once a red block is fully out of the tower interlock and
   resting free on the table, ONE pose write carries it to a spot above the open
   tray — the emulation of the arm picking the free block up and carrying it. The
   final approach into the tray is a gravity drop through the open top, and the
   block must physically settle on the tray floor for the rubric to count it.
3. Success is judged live: both reds resting in the tray + all seven cream blocks
   still in their slots + everything settled.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the scene's
credit is latched), then holds HANDS-OFF for >= 3.5 simulated seconds after
success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS` only if success()
still holds.

Run (forge): python -u -m simgen_tasks.<task>.solve --headless [--seed N]
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
    from . import scene as scene_mod  # noqa: F401  (registers simgen.jenga_quarry)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.jenga_quarry")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    zero_w = torch.zeros(n, 1, 3, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the scene's own readouts.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def local(i: int) -> torch.Tensor:
        return scene.block_local(i)[0]

    def local_dir_to_world(dx: float, dy: float) -> torch.Tensor:
        yaw = float(scene._rig_yaw[0])
        return torch.tensor([dx * math.cos(yaw) - dy * math.sin(yaw),
                             dx * math.sin(yaw) + dy * math.cos(yaw), 0.0], device=device)

    def cream_drift(who: list | None = None) -> float:
        """Max cream-block xy drift from its slot (m) — the preservation telemetry.
        If `who` is given, appends the argmax block index to it."""
        worst, worst_i = 0.0, -1
        for i in range(2, 9):
            d = scene.block_local(i)[0] - scene._rst_local[0, i]
            v = float(d[0:2].norm())
            if v > worst:
                worst, worst_i = v, i
        if who is not None:
            who.append(worst_i)
        return worst

    def report(tag: str) -> None:
        p0, p1 = local(0), local(1)
        print(f"[solve] {tag:14s} | red0_l=({float(p0[0]):+.3f},{float(p0[1]):+.3f},"
              f"{float(p0[2]):.3f}) red1_l=({float(p1[0]):+.3f},{float(p1[1]):+.3f},"
              f"{float(p1[2]):.3f}) standing={bool(scene.tower_standing()[0])} "
              f"drift={cream_drift() * 1000:.1f}mm "
              f"slid={[bool(v) for v in scene._slid[0]]} "
              f"ext={[bool(v) for v in scene._ext[0]]} "
              f"tray={[bool(v) for v in scene._tray_in[0]]} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    last_score = [0.0]

    def print_score(tag: str) -> float:
        sc = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {sc:.4f}", flush=True)
        assert sc >= last_score[0] - 1e-6, \
            f"score decreased across {tag}: {last_score[0]:.4f} -> {sc:.4f}"
        last_score[0] = sc
        return sc

    def settle_wait(max_steps: int = 600, quiet_need: int = 25) -> bool:
        quiet = 0
        for _ in range(max_steps):
            env.step(no_action)
            quiet = quiet + 1 if bool(scene.settled()[0]) else 0
            if quiet >= quiet_need:
                return True
        return False

    def extract(k: int, dx: float, dy: float) -> bool:
        """Contact extraction of red k: velocity-capped axial force at the CoM (the
        fingertip pull on the protruding end) slides it through its channel under
        the roof load until it emerges and drops off the plinth. Weak lateral
        centring keeps it on the slot line. Never touches any other block."""
        block = scene.blocks[k]
        z_slot = float(scene._rst_local[0, k, 2])
        d_w = local_dir_to_world(dx, dy)
        # yaw squaring: ~1 deg of yaw wedges the 13 cm block's corners against its
        # 1 mm-gapped neighbours and geometrically shoves them — keep it square
        # with a small PD torque about z (the wrist's job in a real pull).
        yaw_target = float(scene._rig_yaw[0]) + (0.0 if k == 0 else math.pi / 2)
        # END-GAME LEVEL HOLD. Once the red's CoM passes its last support edge
        # (plinth edge for red0, outermost bottom-layer cream for red1) it
        # seesaws on that edge and its TAIL levers up into the roof layer —
        # that lever is where all the cream drift comes from. A gripped end
        # would simply be held level, so: gravity-overhang feedforward + PD
        # pitch torque keeps it level through the exit; once fully past the
        # support it free-falls LEVEL (tail drops away from the roof cleanly).
        edge = c.plinth_size[0] / 2 if k == 0 else c.span_half
        mgl = c.block_mass * 9.81
        # nose-down rotation axis (world, horizontal, perpendicular to pull)
        ux, uy = float(d_w[0]), float(d_w[1])
        n_axis = torch.tensor([-uy, ux, 0.0], device=device)

        # GRIP WEIGHT SUPPORT (red1 only): a HALF-WEIGHT constant lift.
        # Past ~0.046 travel red1's exit pivot is a CREAM block's top edge:
        # resting the full weight there tips/walks that cream (20 mm drift).
        # But a FULL-weight (0.9 mg) lift is wrong too: the roof block's
        # natural weight share on red1 is ~W/3, and mu_k/mu_s = 0.5 puts the
        # ride-along breakaway exactly there — any real roof press makes the
        # roof block ride red1 1:1 (worst in center-column episodes, where the
        # roof pins red1 and the lift has nowhere to go but into it). A z-servo
        # hover fails the same way: z is pinned, so the servo saturates and its
        # transients press the roof. Half weight splits the difference: the
        # cream edge load AND the worst-case roof press are both halved. The
        # lift cuts once the tail clears the roof (s > 0.118) for a clean level
        # drop. red0 pivots on the kinematic plinth edge (nothing to disturb,
        # proven 0.0 mm drift) — no lift.
        def lift_of(s_along: float, z_now: float) -> float:
            if k == 0 or s_along > 0.118 or z_now < z_slot - 0.003:
                return 0.0
            return 0.5 * mgl * min(max((s_along - 0.030) / 0.020, 0.0), 1.0)

        def control_torque(s_along: float, z_now: float, lift: float) -> torch.Tensor:
            q = block.data.root_quat_w[0]
            om = block.data.root_ang_vel_w[0]
            # yaw squaring: ~1 deg of yaw wedges the 13 cm block's corners
            # against its 1 mm-gapped neighbours — keep it square (wrist job)
            yaw_b = 2.0 * math.atan2(float(q[3]), float(q[0]))
            e = ((yaw_b - yaw_target + math.pi / 2) % math.pi) - math.pi / 2
            tz = max(-0.03, min(0.03, -0.02 * e - 0.002 * float(om[2])))
            # pitch: block long axis (local +x) in world, nose = pull-side end
            qw, qx, qy, qz = (float(v) for v in q)
            ax = 1.0 - 2.0 * (qy * qy + qz * qz)
            ay = 2.0 * (qx * qy + qw * qz)
            az = 2.0 * (qx * qz - qw * qy)
            nose = 1.0 if (ax * ux + ay * uy) >= 0.0 else -1.0
            pitch = az * nose  # >0 = nose up
            wn = float(om @ n_axis)  # >0 = pitching nose-down
            riding = z_now > z_slot - 0.003  # still on its support plane
            ff = -(mgl - lift) * min(max(s_along - edge, 0.0), 0.065) if riding else 0.0
            tn = max(-0.12, min(0.12, ff + 0.4 * pitch - 0.02 * wn))
            t = torch.zeros(n, 1, 3, device=device)
            t[:, 0, 0] = tn * float(n_axis[0])
            t[:, 0, 1] = tn * float(n_axis[1])
            t[:, 0, 2] = tz
            return t
        # JOLT-FREE quasi-static slide. A breakaway slam (escalated force pulse)
        # kicks the roof blocks loose; once a roof block slips, BOTH its
        # interfaces are kinetic and drag == hold, so it rides the red for the
        # rest of the travel. So: ramp the feedforward slowly, brake viscously
        # (push = F - b*v), and cut the feedforward once at breakaway — the red
        # then crawls out at ~1-2 cm/s without ever impulse-kicking the layers
        # above. Stuck/moving is decided by VELOCITY STREAKS (8 steps > 8 mm/s
        # = moving, 30 steps < 4 mm/s = stuck): instantaneous micro-slip noise
        # crosses any single-step threshold and, judged raw, fires the cut over
        # and over, capping F below the true breakaway force.
        F, F_cap, brake, v_cap = 0.4, 6.0, 25.0, 0.020
        ramp = 0.004
        d_l = torch.tensor([dx, dy], device=device)
        mv = st = 0
        moving, armed = False, True
        prog_mark, mark_i = -1.0, 0
        done = False
        for i in range(7000):
            p = local(k)
            hdist = float(p[0:2].norm())
            if hdist > c.ext_dist + 0.010 or float(p[2]) < z_slot - 0.020:
                done = True
                break
            v_along = float(torch.dot(block.data.root_lin_vel_w[0], d_w))
            if v_along > 0.008:
                mv, st = mv + 1, 0
            elif v_along < 0.004:
                mv, st = 0, st + 1
            if not moving and mv >= 8:  # confirmed breakaway
                moving = True
                if armed:  # cut toward the kinetic level, once per stick phase
                    F = max(0.3, F * 0.55)
                    armed = False
            if moving and st >= 30:  # confirmed re-stick: re-arm the cut
                moving, armed = False, True
            if moving and v_along > v_cap:
                F = max(F - 0.02, 0.25)
            elif not (moving and v_along >= v_cap * 0.6):
                F = min(F + ramp, F_cap)  # stuck ramp / under-speed creep
            push = max(0.0, F - brake * max(v_along, 0.0))
            # signed travel along the pull direction (hdist is a norm: nearly
            # flat w.r.t. the first few mm for red0)
            s_along = float((p[0:2] - scene._rst_local[0, k, 0:2]) @ d_l)
            # weak lateral centring toward the slot line (rig-local cross axis)
            cross = float((p[0:2] - scene._rst_local[0, k, 0:2]) @ scene._axis[1 - k])
            lat = max(-0.4, min(0.4, -20.0 * cross * 0.02))
            lift = lift_of(s_along, float(p[2]))
            f_w = d_w * push + local_dir_to_world(
                float(scene._axis[1 - k][0]) * lat, float(scene._axis[1 - k][1]) * lat)
            f_w = f_w + torch.tensor([0.0, 0.0, lift], device=device)
            block.set_external_force_and_torque(
                f_w.view(1, 1, 3).expand(n, 1, 3).contiguous(),
                control_torque(s_along, float(p[2]), lift),
                env_ids=all_ids, is_global=True)
            env.step(no_action)
            if s_along > prog_mark + 0.004:
                prog_mark, mark_i = s_along, i
                who: list = []
                dr = cream_drift(who)
                print(f"[solve] red{k}: s={s_along:+.3f} F={F:.2f} "
                      f"drift={dr * 1000:.1f}mm(b{who[0]})", flush=True)
            elif i - mark_i > 2000:
                print(f"[solve] red{k}: wedged at s={s_along:+.3f} (F={F:.2f})",
                      flush=True)
                break
        block.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        if not done:
            print(f"[solve] red{k}: extraction never cleared the tower", flush=True)
            return False
        # keep nudging it clear along the table until the extraction line, if the
        # fall left it short (still the same axial contact push, now on the ground)
        for i in range(1200):
            p = local(k)
            if float(p[0:2].norm()) > c.ext_dist + 0.010:
                break
            v_along = float(torch.dot(block.data.root_lin_vel_w[0], d_w))
            push = 1.2 if v_along < 0.05 else 0.0
            block.set_external_force_and_torque(
                (d_w * push).view(1, 1, 3).expand(n, 1, 3).contiguous(), zero_w,
                env_ids=all_ids, is_global=True)
            env.step(no_action)
        block.set_external_force_and_torque(zero_w, zero_w, env_ids=all_ids)
        settle_wait(400)
        p = local(k)
        print(f"[solve] red{k}: extracted, local=({float(p[0]):+.3f},"
              f"{float(p[1]):+.3f},{float(p[2]):.3f}) drift={cream_drift() * 1000:.1f}mm",
              flush=True)
        return True

    def bank(k: int, x_off: float) -> None:
        """TRANSPORT ONLY: one pose write carries the free, already-extracted red
        block to above the open tray (the arm's pick-and-carry); the descent into
        the tray is a gravity drop through the open top."""
        p = local(k)
        assert float(p[0:2].norm()) > c.ext_dist, f"red{k} not clear of the tower yet"
        assert float(p[2]) < 0.05, f"red{k} not resting free on the table"
        yaw = float(scene._rig_yaw[0]) + math.pi / 2  # length along tray-local y
        pos_l = torch.tensor([[float(scene._tray_xy[0, 0]) + x_off,
                               float(scene._tray_xy[0, 1]), 0.10]], device=device)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.to_world(pos_l, all_ids) + scene.env_origins
        st[:, 3] = math.cos(yaw / 2)
        st[:, 6] = math.sin(yaw / 2)
        scene.blocks[k].write_root_state_to_sim(st, all_ids)
        settle_wait(400)
        p = local(k)
        print(f"[solve] red{k}: banked, local=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):.3f}) in_tray={bool(scene.in_tray(k)[0])}", flush=True)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(60)
    s0, s1 = int(scene._slot[0, 0]), int(scene._slot[0, 1])
    print(f"[solve] layout readback (seed {args.seed}): "
          f"rig_xy=({float(scene._rig_xy[0, 0]):+.3f},{float(scene._rig_xy[0, 1]):+.3f}) "
          f"rig_yaw={math.degrees(float(scene._rig_yaw[0])):+.1f} deg "
          f"red slots=(L0:{s0}, L1:{s1}) "
          f"tray_l=({float(scene._tray_xy[0, 0]):+.3f},{float(scene._tray_xy[0, 1]):+.3f})",
          flush=True)
    report("reset")
    p0 = local(0)
    assert abs(float(p0[2]) - c.layer_z(0)) < 0.006, "red0 not at bottom-layer height"
    assert bool(scene.tower_standing()[0]), "tower not standing at reset"
    assert not bool(scene.success()[0]), "success at reset (broken)"
    sc = print_score("P0 reset+settle")
    assert sc <= 0.02, f"reset score should be ~0, got {sc}"

    # ---------------- phase 1: Jenga-slide red0 out of the bottom layer --------------------
    # direction -x: away from the tray, so the falling block can never land near it
    if not extract(0, -1.0, 0.0):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (red0 extraction wedged)", flush=True)
        os._exit(1)
    report("extract0")
    assert bool(scene.tower_standing()[0]), "tower disturbed by red0 extraction"
    print_score("P1 red0 slid out under the roof load")

    # ---------------- phase 2: carry red0 to the tray (drop through the open top) ----------
    bank(0, -0.040)
    report("bank0")
    print_score("P2 red0 dropped into the tray")

    # ---------------- phase 3: Jenga-slide red1 out of the middle layer --------------------
    if not extract(1, 0.0, 1.0):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (red1 extraction wedged)", flush=True)
        os._exit(1)
    report("extract1")
    assert bool(scene.tower_standing()[0]), "tower disturbed by red1 extraction"
    print_score("P3 red1 slid out under the roof load")

    # ---------------- phase 4: carry red1 to the tray --------------------------------------
    bank(1, 0.040)
    settle_wait(600)
    report("bank1")
    if not bool(scene.success()[0]):
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (goal state not reached)", flush=True)
        os._exit(1)
    s4 = print_score("P4 both reds banked, tower standing, success verified")

    # ---------------- phase 5: persistence (>= 3.5 simulated seconds, no intervention) -----
    hold = True
    for _ in range(10):  # 10 x 42 steps = 420 substeps = 3.5 s at 120 Hz
        step(42)
        hold = hold and bool(scene.success()[0])
    report("persist")
    s5 = print_score("P5 persistence 3.5 s")
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
    except Exception as exc:  # noqa: BLE001 — die fast, not at the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

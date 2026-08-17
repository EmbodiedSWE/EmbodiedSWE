"""solve — teleport solution for PortDropboxScene (…tomato_sauce…basket_i203).

Scene-level env (robot="null"). Teleports handle TRANSPORT ONLY: the single pose
write on the red can carries it from its ground spawn to a free rest pose LYING on
the loading tray, fully OUTSIDE the port tunnel (its nose 30 mm short of the outer
wall face) — the stand-in for the Franka pick + reorient + set-down of TASK.md.
Every load-bearing interaction runs through contact dynamics:

  - STAGE: the teleported can free-falls 3 mm onto the tray and RESTS between the
    rails (the staged latch is slow-gated — it only sets on the rested state);
  - INSERT: a velocity-regulated horizontal push force at the can's CoM, aimed
    along the box's port axis (bang-bang: F while axial speed < 0.12 m/s, +1.5 N
    per 1.5 s of stall, 3.5..9 N — fingertip-scale authority on a 350 g can),
    slides the can across the tray, over the sill and through the port tunnel.
    Rails, jambs, sill and lintel act on it the whole way. The force cuts once
    the CoM is 12 mm past the inner wall face (10 mm beyond the rubric's
    commitment line, covering the rubric's one-step sampling lag) — from there
    the can TIPS and gravity drops it to the box floor unaided;
  - the beige distractor is never touched.

Prints `SIM_GEN_SCORE <v>` at phase boundaries (asserted non-decreasing: latched
credit must not evaporate). After success() first holds, keeps simulating >= 3.5
more simulated seconds with the push buffer zero (asserted); only if success()
still holds prints exactly `SIM_GEN_SOLVE: SUCCESS`. Hard exit (os._exit) after
the verdict, watchdog Timer as backstop — Kit teardown hangs otherwise.

Run (forge): python -u -m simgen_tasks.living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i203.solve --headless [--seed N]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_sec", type=float, default=600.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.living_room_scene1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_i203 \
        import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SOLVE: TIMEOUT", flush=True),
                                            os._exit(3)))
wd.daemon = True
wd.start()

# Push plant (fingertip authority on the 350 g can; sliding friction ~1.7 N).
F0 = 3.5  # N starting push
F_MAX = 9.0  # N escalation ceiling
F_STEP = 1.5  # N per stall escalation
V_CAP = 0.12  # m/s: push only while slower (quasi-static — no ballistic vaulting)
STALL_STEPS = 180  # 1.5 s windows for the stall watch
STALL_EPS = 0.003  # m minimum advance per window


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.port_dropbox")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def sc() -> float:
        return float(scene.score()[0])

    def local_red() -> torch.Tensor:
        return scene.bin_local(scene.red.data.root_pos_w)[0]

    def report(tag: str) -> None:
        p = local_red()
        print(f"[solve] {tag:12s} red_local=({float(p[0]):+.3f}, {float(p[1]):+.3f}, "
              f"{float(p[2]):.3f}) staged={float(scene.staged_latch[0]):.2f} "
              f"insert={float(scene.insert_latch[0]):.2f} "
              f"inside={bool(scene.inside(scene.red)[0])} "
              f"still={int(scene.still_streak[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])}", flush=True)

    last_score = -1.0

    def phase_score(tag: str) -> None:
        nonlocal last_score
        s = sc()
        assert s >= last_score - 1e-6, f"score decreased at {tag}: {last_score} -> {s}"
        last_score = s
        print(f"SIM_GEN_SCORE {s:.3f}", flush=True)

    def verdict(ok: bool) -> None:
        print("SIM_GEN_SOLVE: SUCCESS" if ok else "SIM_GEN_SOLVE: FAIL", flush=True)
        code = 0 if ok else 1
        t = threading.Timer(10.0, lambda: os._exit(code))
        t.daemon = True
        t.start()
        try:
            env.close()
            app.close()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    # ================= P0: reset + settle + layout readback ====================================
    env.reset(seed=args.seed)
    step(60)
    yaw = float(scene.bin_yaw[0])
    bx, by = (float(v) for v in scene.bin_xy[0])
    rx, ry = (float(v) for v in scene.red0[0])
    gx, gy = (float(v) for v in scene.beige0[0])
    print(f"[solve] seed={args.seed} bin=({bx:+.3f}, {by:+.3f}) yaw={math.degrees(yaw):+.1f}deg "
          f"red=({rx:+.3f}, {ry:+.3f}) beige=({gx:+.3f}, {gy:+.3f})", flush=True)
    report("reset")
    assert float(scene.push_f.abs().max()) == 0.0
    assert sc() < 0.05, "null reset must score ~0"
    phase_score("reset")  # ~0.000

    # ================= P1: TRANSPORT (the only pose write on the red can) ======================
    # Lying on the tray, axis along the port axis, nose 30 mm OUTSIDE the outer wall
    # face, 3 mm above the tray top -> free drop, rest, slow-gated staged latch.
    cy, sy = math.cos(yaw), math.sin(yaw)
    lx, ly, lz = c.tray_cx, 0.0, c.can_stage_z + 0.003
    origin = scene.env_origins[0]
    st = torch.zeros(1, 13, device=device)
    st[0, 0] = origin[0] + bx + lx * cy - ly * sy
    st[0, 1] = origin[1] + by + lx * sy + ly * cy
    st[0, 2] = origin[2] + lz
    half = yaw / 2
    c45 = math.cos(math.pi / 4)
    st[0, 3] = math.cos(half) * c45
    st[0, 4] = -math.sin(half) * c45
    st[0, 5] = math.cos(half) * c45
    st[0, 6] = math.sin(half) * c45
    scene.red.write_root_state_to_sim(st, torch.tensor([0], device=device))
    step(90)  # drop 3 mm + rest (staged latch is slow-gated)
    report("staged")
    assert float(scene.staged_latch[0]) > 0.99, "staged latch must set on the rested state"
    phase_score("staged")  # ~0.250

    # ================= P2: INSERT through the port (contact dynamics) ==========================
    dirx, diry = cy, sy  # bin-local +x (tray -> box) in world
    force = F0
    p0 = float(local_red()[0])
    last_x, last_ck = p0, 0
    committed = False
    for i in range(2400):  # 20 s budget
        p = local_red()
        # Cut 10 mm beyond the rubric's commitment line (com_full = inner face + 2 mm):
        # the rubric's post_step samples lag a step, so push a hair past to guarantee
        # a gated full-progress sample before the tipping pitch breaks the lying gate.
        if float(p[0]) > c.inner_face + 0.012:
            committed = True
            break
        v = scene.red.data.root_lin_vel_w[0]
        v_ax = float(v[0]) * dirx + float(v[1]) * diry
        f = force if v_ax < V_CAP else 0.0
        scene.push_f[0, 0, 0] = f * dirx
        scene.push_f[0, 0, 1] = f * diry
        step(1)
        if i - last_ck >= STALL_STEPS:  # stall watch: escalate the force
            x_now = float(local_red()[0])
            if x_now - last_x < STALL_EPS and force < F_MAX:
                force = min(F_MAX, force + F_STEP)
                print(f"[solve] push stalled at x={x_now:+.3f} -> F={force:.1f} N", flush=True)
            last_x, last_ck = x_now, i
        if i and i % 240 == 0:
            report(f"push{i}")
    scene.push_f[0, 0, :] = 0.0
    report("push-off")
    if not committed:
        print("[solve] PHASE 2 FAILED: CoM never crossed the commitment line", flush=True)
        verdict(False)

    # Hands off: the can tips through the port and falls to the box floor by gravity.
    # (The insert latch tops out on these first hands-off steps: the rubric's post_step
    # sees each state one step late, so it latches full progress just after the break.)
    ok = False
    for _ in range(96):  # up to 8 s for inside + settled
        if bool(scene.success()[0]):
            ok = True
            break
        step(10)
    if not ok:
        report("drop-fail")
        print("[solve] PHASE 2 FAILED: success() not reached after the drop", flush=True)
        verdict(False)
    step(60)  # bed down
    report("inside")
    assert float(scene.insert_latch[0]) > 0.99, "insert latch must be full after commitment"
    phase_score("inserted")  # 1.000

    # ================= P3: persistence (>= 3.5 simulated seconds, hands off) ===================
    assert float(scene.push_f.abs().max()) == 0.0, "push buffer must be zero for persistence"
    persist = int(round(3.5 / env.dt))
    step(persist)
    report("final")
    phase_score("final")
    still_ok = bool(scene.success()[0]) and abs(sc() - 1.0) < 1e-3
    print(f"[solve] persistence: {persist} steps ({persist * env.dt:.2f} s) hands-off, "
          f"success={still_ok}", flush=True)
    verdict(still_ok)


if __name__ == "__main__":
    main()

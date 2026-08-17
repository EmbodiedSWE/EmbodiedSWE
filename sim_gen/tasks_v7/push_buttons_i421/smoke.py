"""smoke — REJECTION battery for the DropSequencerScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py already proves the rubric ACCEPTS the correct
outcome on two seeds with different permutations). Every check here CONSTRUCTS a wrong
outcome as a settled state (teleports are instrumentation; probe forces go through the
scene's frame-encoded world-force buffers) and asserts the rubric REJECTS it:

  1. settle/no-NaN   — reset settles finite: plungers at OUT rest, every marble parked on
                       its perch, nothing dropped, score ~0;
  2. randomization   — READBACK: vault xy + yaw vary across seeded resets; the marble ->
                       chimney permutation varies (>= 3 distinct of 6 draws) and each
                       marble's chimney-x matches the chim readback every time;
  3. null policy     — 600 idle steps -> nothing drops, score ~0, no success;
  4. weak press      — a ramped 0.5 N press (about half the drop threshold) is a real
                       probe (>= 1.5 mm travel) but drops nothing; releases home, score 0;
  5. wrong first drop — the seed strategy (press all the buttons, wrong order): azure's
                       chimney pressed FIRST, then crimson's, then amber's — every press
                       real, all three marbles queued and settled, plungers home — yet the
                       wrong marble at the stop caps the episode at the 0.10 drop latch
                       forever, no success;
  6. mid-way wrong drop — crimson's chimney first (banks the 0.35 prefix-1 latch), then
                       azure's out of order, then amber's: score stays latched at 0.35, no
                       success — partial credit is kept, the rest is permanently lost;
  7. missing third   — crimson + amber constructed correctly queued, azure still on its
                       perch: prefix 2 (0.60), no success;
  8. outside the vault — all three settled on the GROUND beyond the stop in the correct
                       left-to-right order: order outside the roofed queue zone counts
                       nothing beyond the 0.10 drop latch, no success;
  9. plunger-home gate — a correct queue constructed while one plunger is HELD at its hard
                       stop: hold counter stays 0, no success for 6 s; releasing the
                       plunger (sanctioned segment) is accepted -> success, score 1.0;
 10. roof captivity  — a 0.5 N upward pull on the queued stop marble rises to the roof and
                       STOPS (probe is real: it rose), the marble cannot leave the channel;
 11. no reordering   — every marble servo-driven 30+ mm back up the channel (probe is
                       real); success reverts LIVE while displaced; the marbles can never
                       pass one another (the 44 mm channel keeps x-crossover geometrically
                       impossible); released, everything rolls back and re-queues in the
                       SAME order -> success returns (still the sanctioned segment);
 12. no accidental success — success() never fired outside the sanctioned segment;
 13. final no-NaN    — every body finite at the end.

Records video frames -> frames.npz in the CWD.
Run (forge): python -u -m simgen_tasks.push_buttons_i421.smoke --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--max_sec", type=float, default=1200.0)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.push_buttons_i421 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

G = scene_mod.G
NAMES = scene_mod.MARBLE_NAMES

_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drop_sequencer")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    ex = torch.zeros(1, 3, device=device)
    ex[0, 0] = 1.0

    # --- recording (viewport rgb annotator, the proven forge mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.55, -0.85, 0.72)) + o),
                                tuple(np.array((0.05, 0.0, 0.10)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0
    saw_success = False
    allow_success = False

    def step(k: int) -> None:
        nonlocal step_i, saw_success
        for _ in range(k):
            env.step(no_action)
            if not allow_success:
                saw_success = saw_success or bool(scene.success()[0])
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        q = scene.rod_q()[0]
        rel = scene.marble_rel()[0]
        print(f"[smoke] {tag:14s} q=({q[0] * 1000:5.1f},{q[1] * 1000:5.1f},"
              f"{q[2] * 1000:5.1f})mm "
              + " ".join(f"{nm[:2]}=({rel[i, 0] * 1000:+6.1f},{rel[i, 2] * 1000:5.1f})"
                         for i, nm in enumerate(NAMES))
              + f" drop={int(scene._dropped[0, 0])}{int(scene._dropped[0, 1])}"
                f"{int(scene._dropped[0, 2])} p={int(scene.queue_prefix_now()[0])}"
                f"/{int(scene._prefix[0])} score={sc():.3f} "
                f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def push_off() -> None:
        for k in scene.push_w:
            scene.push_w[k].zero_()

    def place(body, local_xyz) -> None:
        """Teleport a body to a vault-frame pose (vault quat), zero velocity."""
        pv = scene.vault.data.root_pos_w[0]
        qv = scene.vault.data.root_quat_w[0]
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = pv + quat_apply(qv.view(1, 4),
                                     torch.tensor([local_xyz], device=device))[0]
        st[0, 3:7] = qv
        body.write_root_state_to_sim(st, ids)

    def zq(x: float) -> float:
        """Marble rest center height on the channel floor at channel-x, +1.5 mm."""
        return G.FZ0 + x * math.tan(G.SLOPE) + G.R + 0.0015

    def on_perch(mi: int) -> bool:
        rel = scene.marble_rel()[0, mi]
        return (abs(float(rel[1]) - (-0.012)) < 0.006 and 0.135 <= float(rel[2]) <= 0.155
                and abs(float(rel[0]) - G.HX[int(scene.chim[0, mi])]) < 0.006)

    def rods_home() -> bool:
        return bool((scene.rod_q()[0] < c.rod_home).all())

    def press_drop(rod: int, mi: int) -> bool:
        """Solve's audited velocity-servo press to the hard stop; True iff marble mi
        dropped. Gain audit: 6*dt/m = 6/(120*0.06) = 0.83 < 1."""
        key = f"rod{rod}"
        dropped, at = False, 0
        for _ in range(900):
            axis = scene.press_axis_w()[0]
            v = float((scene.rods[rod].data.root_lin_vel_w[0] * axis).sum())
            q = float(scene.rod_q()[0, rod])
            v_des = 0.06 if at < 8 else 0.0
            f = max(0.0, min(4.5, 6.0 * (v_des - v) + c.spring_k * max(0.0, q) + 0.6))
            scene.push_w[key][0] = axis * f
            step(1)
            if float(scene.rod_q()[0, rod]) >= G.STROKE - 0.0025:
                at += 1
            else:
                at = max(0, at - 1)
            if bool(scene._dropped[0, mi]):
                dropped = True
                if at >= 8:
                    break
        scene.push_w[key][0] = 0.0
        return dropped

    def hold_press(rod: int, n: int) -> None:
        """Continuous servo hold of a plunger at its hard stop for n substeps."""
        key = f"rod{rod}"
        for _ in range(n):
            axis = scene.press_axis_w()[0]
            v = float((scene.rods[rod].data.root_lin_vel_w[0] * axis).sum())
            q = float(scene.rod_q()[0, rod])
            v_des = 0.06 if q < G.STROKE - 0.0025 else 0.0
            f = max(0.0, min(4.5, 6.0 * (v_des - v) + c.spring_k * max(0.0, q) + 0.6))
            scene.push_w[key][0] = axis * f
            step(1)

    def wait_settled(mis: list[int]) -> bool:
        """Hands off: wait until every listed marble is in the queue zone and still."""
        for _ in range(90):
            step(10)
            inq = scene.in_queue()[0]
            still = scene.marbles_still()[0]
            if all(bool(inq[i]) and bool(still[i]) for i in mis):
                return True
        return False

    def qx(mi: int) -> float:
        return float(scene.marble_rel()[0, mi, 0])

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (scene.vault, *scene.rods, *scene.marbles))
    check("settle: finite, plungers at OUT rest, every marble parked on its perch, "
          "nothing dropped, score ~0",
          fin and rods_home() and all(on_perch(i) for i in range(3))
          and not bool(scene._dropped[0].any())
          and sc() <= 0.005 and not bool(scene.success()[0]))

    # =========================== 2. randomization is real ===================================
    reads, perms = [], set()
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        kp = (scene.vault.data.root_pos_w - scene.env_origins)[0]
        q = scene.vault.data.root_quat_w[0]
        yaw = float(torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                1 - 2 * (q[2] * q[2] + q[3] * q[3])))
        perm = tuple(int(v) for v in scene.chim[0])
        perms.add(perm)
        ok = all(on_perch(i) for i in range(3)) and rods_home()
        reads.append((float(kp[0]), float(kp[1]), yaw, ok))
        print(f"[smoke] seed {sd}: vault=({kp[0]:+.3f},{kp[1]:+.3f}) yaw={yaw:+.2f} "
              f"chim={perm} assembled={ok}", flush=True)
    arr = np.array([r[:3] for r in reads])
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: vault xy + yaw vary (readback), >= 3 distinct marble->chimney "
          "permutations of 6 draws, marbles parked on the chimney the chim readback "
          "names every reset",
          spread[0] > 0.05 and spread[1] > 0.05 and spread[2] > 1.0
          and len(perms) >= 3 and all(r[3] for r in reads))

    # =========================== 3. null policy =============================================
    env.reset(seed=31)
    step(600)
    report("null-policy")
    check("null policy: 600 idle steps -> nothing drops, marbles stay perched, score ~0, "
          "no success",
          not bool(scene._dropped[0].any()) and all(on_perch(i) for i in range(3))
          and sc() <= 0.005 and not bool(scene.success()[0]))

    # =========================== 4. weak press drops nothing ================================
    env.reset(seed=41)
    step(60)
    rod = int(scene.chim[0, 0])  # crimson's chimney
    q_peak = -1.0
    for i in range(180):
        f = 0.5 * min(1.0, (i + 1) / 60)  # ramped to 0.5 N ~ half the drop threshold
        scene.push_w[f"rod{rod}"][0] = scene.press_axis_w()[0] * f
        step(1)
        q_peak = max(q_peak, float(scene.rod_q()[0, rod]))
    scene.push_w[f"rod{rod}"][0] = 0.0
    step(120)
    report("weak-press")
    check("weak press: ramped 0.5 N moves the plunger (probe is real: >= 1.5 mm) but "
          "below the drop threshold — no marble drops, plunger returns home, score 0",
          0.0015 <= q_peak <= 0.0055 and not bool(scene._dropped[0].any())
          and rods_home() and all(on_perch(i) for i in range(3)) and sc() <= 0.005)

    # =========================== 5. wrong first drop (seed strategy) ========================
    env.reset(seed=51)
    step(60)
    chim = [int(v) for v in scene.chim[0]]
    d3 = press_drop(chim[2], 2)  # azure FIRST: permanently wrong
    s3 = wait_settled([2])
    mid = sc()
    d1 = press_drop(chim[0], 0)
    s1 = wait_settled([0, 2])
    d2 = press_drop(chim[1], 1)
    s2 = wait_settled([0, 1, 2])
    step(60)
    report("wrong-first")
    check("wrong first drop: all three chimneys really pressed (azure's first) and all "
          "three marbles queued + settled + plungers home — the wrong stop marble caps "
          "the episode at the 0.10 drop latch, prefix 0, no success",
          d3 and s3 and d1 and s1 and d2 and s2 and rods_home()
          and 0.095 <= mid <= 0.105 and 0.095 <= sc() <= 0.105
          and qx(2) < qx(0) < qx(1) and int(scene._prefix[0]) == 0
          and not bool(scene.success()[0]))

    # =========================== 6. mid-way wrong drop ======================================
    env.reset(seed=61)
    step(60)
    chim = [int(v) for v in scene.chim[0]]
    dA = press_drop(chim[0], 0)  # crimson first: correct, banks prefix 1
    sA = wait_settled([0])
    step(60)
    bankA = sc()
    dB = press_drop(chim[2], 2)  # azure second: wrong
    sB = wait_settled([0, 2])
    dC = press_drop(chim[1], 1)
    sC = wait_settled([0, 1, 2])
    step(60)
    report("mid-wrong")
    check("mid-way wrong drop: crimson first banks the 0.35 prefix-1 latch; azure out of "
          "order caps it there forever — final settled queue, plungers home, score still "
          "0.35, no success",
          dA and sA and dB and sB and dC and sC and rods_home()
          and 0.34 <= bankA <= 0.36 and 0.34 <= sc() <= 0.36
          and qx(0) < qx(2) < qx(1) and int(scene._prefix[0]) == 1
          and not bool(scene.success()[0]))

    # =========================== 7. missing third marble ====================================
    env.reset(seed=71)
    step(60)
    place(scene.marbles[0], (0.013, 0.0, zq(0.013)))
    place(scene.marbles[1], (0.040, 0.0, zq(0.040)))
    step(300)
    report("missing-third")
    check("missing third: crimson + amber constructed correctly queued, azure still on "
          "its perch -> prefix 2 (0.60 band), no success",
          int(scene.queue_prefix_now()[0]) == 2 and on_perch(2)
          and 0.59 <= sc() <= 0.61 and rods_home() and not bool(scene.success()[0]))

    # =========================== 8. correct order OUTSIDE the vault =========================
    env.reset(seed=81)
    step(60)
    for mi, x in enumerate((-0.05, -0.09, -0.13)):
        place(scene.marbles[mi], (x, 0.0, G.R + 0.002))
    step(300)
    report("outside")
    inq = scene.in_queue()[0]
    check("outside the vault: all three settled on the ground beyond the stop in the "
          "correct left-to-right order — order outside the roofed queue zone counts "
          "nothing beyond the 0.10 drop latch, no success",
          not bool(inq.any()) and int(scene.queue_prefix_now()[0]) == 0
          and 0.095 <= sc() <= 0.105 and not bool(scene.success()[0]))

    # =========================== 9. plunger-home gate + sanctioned acceptance ===============
    env.reset(seed=91)
    step(60)
    place(scene.marbles[0], (0.013, 0.0, zq(0.013)))
    place(scene.marbles[1], (0.038, 0.0, zq(0.038)))
    place(scene.marbles[2], (0.063, 0.0, zq(0.063)))
    hold_press(0, 720)  # 6 s held at the hard stop while the correct queue settles
    held_ok = (int(scene.queue_prefix_now()[0]) == 3 and int(scene._hold[0]) == 0
               and float(scene.rod_q()[0, 0]) >= G.STROKE - 0.003
               and not bool(scene.success()[0]) and 0.84 <= sc() <= 0.86)
    report("held-rod")
    allow_success = True  # sanctioned segment: release -> genuine accept
    scene.push_w["rod0"][0] = 0.0
    got = False
    for _ in range(100):
        step(10)
        if bool(scene.success()[0]):
            got = True
            break
    report("released")
    check("plunger-home gate: a correct settled queue with one plunger HELD at its stop "
          "earns no success for 6 s (hold counter pinned 0); releasing it is accepted -> "
          "success, score 1.0",
          held_ok and got and sc() == 1.0)

    # =========================== 10. roof captivity (lift blocked) ==========================
    z0 = float(scene.marble_rel()[0, 0, 2])
    z_peak, in_ch = -1.0, True
    up = torch.zeros(1, 3, device=device)
    up[0, 2] = 1.0
    for _ in range(120):
        scene.push_w["crimson"][0] = up[0] * 0.5  # 2.5x marble weight, straight up
        step(1)
        z_peak = max(z_peak, float(scene.marble_rel()[0, 0, 2]))
        in_ch = in_ch and bool(scene.in_channel()[0, 0])
    push_off()
    step(240)
    report("lift-tried")
    check("roof captivity: a 0.5 N upward pull (2.5x weight) on the stop marble rises to "
          "the roof and stops (probe is real: it rose), marble never leaves the channel",
          z_peak >= z0 + 0.003 and z_peak <= G.ROOF_Z0 - G.R + 0.003 and in_ch)

    # =========================== 11. no reordering (drive the train up-slope) ===============
    # The settled queue is STAGGERED (x-gaps ~12.5 mm, y zigzag: max |dy| between centers
    # is 20 mm < 24 mm, so x-crossover — passing — is geometrically impossible; the
    # staggered-contact minimum x-gap is sqrt(24^2 - 20^2) ~ 13.3 mm). Drive each marble
    # individually (pushing only the front one wedges the staggered train on the walls).
    xhat = quat_apply(scene.vault.data.root_quat_w, ex)[0]
    x0 = qx(0)
    reverted, in_ch, min_gap = False, True, 1.0
    for _ in range(500):
        xc = qx(0)
        if xc >= 0.045:
            break
        for mi, nm in enumerate(NAMES):
            v = float((scene.marbles[mi].data.root_lin_vel_w[0] * xhat).sum())
            # gain 2.0: 2*dt/m = 2/(120*0.02) = 0.83 < 1; cap 0.8 N
            scene.push_w[nm][0] = xhat * max(-0.8, min(0.8, 2.0 * (0.08 - v) + 0.10))
        step(1)
        in_ch = in_ch and bool(scene.in_channel()[0].all())
        min_gap = min(min_gap, qx(1) - qx(0), qx(2) - qx(1))
        if xc > x0 + 0.010:
            reverted = reverted or (not bool(scene.success()[0]))
    push_off()
    moved = qx(0) - x0
    report("shoved")
    back = False
    for _ in range(120):
        step(10)
        if bool(scene.success()[0]):
            back = True
            break
    report("rolled-back")
    check("no reordering: the whole train driven 30+ mm back up the channel (success "
          "reverts LIVE while displaced), marbles never pass one another (x order and "
          "the staggered-contact minimum gap held) or leave the channel; released, they "
          "re-queue in the same order -> success returns",
          moved >= 0.030 and reverted and in_ch and min_gap >= 0.010
          and back and qx(0) < qx(1) < qx(2) and sc() == 1.0)
    allow_success = False

    # =========================== 12./13. global =============================================
    check("no accidental success: success() never fired outside the sanctioned segment",
          not saw_success)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (scene.vault, *scene.rods, *scene.marbles))
    check("final: every body state finite", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drop_sequencer")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, ok in checks:
            if not ok:
                print(f"[smoke]   FAILED: {nm}", flush=True)
    code = 0 if all_ok else 1
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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL", flush=True)
        os._exit(1)

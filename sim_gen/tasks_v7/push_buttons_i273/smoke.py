"""smoke — REJECTION battery for the ShutterTrapScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py already proves the rubric ACCEPTS the correct
outcome on three seeds, both corridor ends). Every check here CONSTRUCTS a wrong outcome
as a settled state (teleports are instrumentation; forces go through the scene's
frame-encoded probe buffers) and asserts the rubric REJECTS it:

  1. settle/no-NaN   — reset settles finite: plungers at UP rest, plate parked at its end,
                       nothing covered, score ~0;
  2. randomization   — READBACK: console xy/yaw, the plate's corridor END (both sides seen)
                       and its start gap differ across seeded resets; plungers assembled at
                       every spawn;
  3. null policy     — 600 idle steps -> score ~0, plate parked, no success;
  4. seed strategy   — the seed's whole plan (press the three buttons, any order): all
                       three pressed to the hard stop (probe is real: 20+ mm reached each)
                       with the shutter untouched -> every head creeps back up, ends
                       fully popped, score stays in the 0.05 press band, no success;
  5. held press      — holding one button at the bottom for 2 s (never chasing with the
                       plate) banks nothing: no trap, press-band score, no success; released
                       -> pops;
  6. shove at raised head — ramming the plate (ramped 6 N) at an UN-pressed button: the
                       plate advances to the head and STALLS against it (probe is real:
                       it moved, then held still under sustained force), does not climb,
                       does not press the head, banks nothing;
  7. lift-out        — a 12 N vertical pull takes up the 2 mm lip clearance and STOPS: the
                       plate cannot leave the corridor (probe is real: it rose to the lips);
  8. late chase      — press-release, wait ~7 s (button fully popped), THEN push: blocked
                       exactly like 6; only the 0.05 press latch remains, no trap credit;
  9. partial coverage — all three plungers constructed pressed with the plate physically
                       overhanging all three but 6 mm outside the judged all-covered
                       window: two heads judged trapped (0.55 band), the third is
                       physically held yet judged uncovered -> success stays False;
 10. pry-back        — from a CONSTRUCTED full success (accepted, sanctioned segment), the
                       plate is pried back toward its end: uncovered heads pop back up,
                       success reverts to False live, the trapped_ever latches keep 0.80 —
                       success must be held by the physical trap, not banked;
 11. no accidental success — success() never fired outside the sanctioned segment;
 12. final no-NaN    — every body finite at the end.

Records video frames -> frames.npz in the CWD.
Run (forge): python -u -m simgen_tasks.push_buttons_i273.smoke --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=8)
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

import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
from simgen_tasks.push_buttons_i273 import scene as scene_mod  # noqa: E402

G = scene_mod.G

_wd = threading.Timer(args.max_sec, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                             os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shutter_trap")().build(num_envs=1, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.70, -1.30, 0.95)) + o),
                                tuple(np.array((0.0, 0.0, 0.07)) + o),
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

    def report(tag: str) -> None:
        d = scene.depth()[0]
        cov = scene.covered()[0]
        print(f"[smoke] {tag:16s} d=({d[0] * 1000:5.1f},{d[1] * 1000:5.1f},"
              f"{d[2] * 1000:5.1f})mm xp={float(scene.plate_x()[0]) * 1000:+7.1f}mm "
              f"cov={int(cov[0])}{int(cov[1])}{int(cov[2])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def push_off() -> None:
        for k in scene.push_w:
            scene.push_w[k].zero_()

    def place(body, local_xyz, settle: int = 0) -> None:
        """Teleport a body to a console-frame pose (console quat), zero velocity."""
        pc = scene.console.data.root_pos_w[0]
        qc = scene.console.data.root_quat_w[0]
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = pc + quat_apply(qc.view(1, 4),
                                     torch.tensor([local_xyz], device=device))[0]
        st[0, 3:7] = qc
        body.write_root_state_to_sim(st, ids)
        if settle:
            step(settle)

    def press_full(i: int, hold: int = 10) -> float:
        """Velocity-servo press of plunger i to the hard stop; returns the peak depth."""
        key, peak, at = f"p{i}", -1.0, 0
        for _ in range(600):
            up = scene.up_w()[0]
            v_dn = float(-(scene.plungers[i].data.root_lin_vel_w[0] * up).sum())
            scene.push_w[key][0] = -up * max(0.0, min(3.5, 20.0 * (0.08 - v_dn) + 1.0))
            step(1)
            peak = max(peak, float(scene.depth()[0, i]))
            if float(scene.depth()[0, i]) >= G.D_MAX - 0.0015:
                at += 1
                if at >= 10:
                    break
            else:
                at = 0
        for _ in range(hold):
            up = scene.up_w()[0]
            v_dn = float(-(scene.plungers[i].data.root_lin_vel_w[0] * up).sum())
            scene.push_w[key][0] = -up * max(0.0, min(3.5, 20.0 * (0.0 - v_dn) + 1.0))
            step(1)
            peak = max(peak, float(scene.depth()[0, i]))
        scene.push_w[key][0] = 0.0
        return peak

    def chase_to(t_x: float, s: float, n_max: int = 500) -> None:
        """Velocity-servo plate push to console-frame x target (solve's audited gains)."""
        u = -s * quat_apply(scene.console.data.root_quat_w, ex)[0]
        for _ in range(n_max):
            dist = (float(scene.plate_x()[0]) - t_x) * s
            if dist <= 0.003:
                break
            v_des = max(0.02, min(0.11, 4.0 * dist))
            v = float((scene.plate.data.root_lin_vel_w[0] * u).sum())
            scene.push_w["plate"][0] = u * max(-4.0, min(4.0, 40.0 * (v_des - v) + 0.5))
            step(1)
        for _ in range(20):
            v = float((scene.plate.data.root_lin_vel_w[0] * u).sum())
            scene.push_w["plate"][0] = u * max(-4.0, min(4.0, 40.0 * (0.0 - v)))
            step(1)
        scene.push_w["plate"][0] = 0.0

    def side() -> float:
        return 1.0 if float(scene.plate_x()[0]) > 0 else -1.0

    def shove_at_head(n_ramp: int = 120, n_hold: int = 240) -> tuple[float, float, float]:
        """Ramped 6 N shove toward the corridor centre; returns (moved, |xp|_end, |v|_end)."""
        s = side()
        u = -s * quat_apply(scene.console.data.root_quat_w, ex)[0]
        x0 = abs(float(scene.plate_x()[0]))
        for i in range(n_ramp + n_hold):
            scene.push_w["plate"][0] = u * (6.0 * min(1.0, (i + 1) / n_ramp))
            step(1)
        v_end = float(scene.plate.data.root_lin_vel_w[0].norm())
        x_end = abs(float(scene.plate_x()[0]))
        scene.push_w["plate"][0] = 0.0
        return x0 - x_end, x_end, v_end

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(90)
    report("reset")
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (scene.console, scene.plate, *scene.plungers))
    d = scene.depth()[0]
    check("settle: finite, plungers at UP rest, plate parked at its end, nothing "
          "covered, score ~0",
          fin and bool((d.abs() < 0.003).all())
          and 0.200 <= abs(float(scene.plate_x()[0])) <= 0.222
          and not bool(scene.covered()[0].any())
          and float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0]))

    # =========================== 2. randomization is real ===================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26):
        env.reset(seed=sd)
        step(30)
        kp = (scene.console.data.root_pos_w - scene.env_origins)[0]
        q = scene.console.data.root_quat_w[0]
        yaw = float(torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                                1 - 2 * (q[2] * q[2] + q[3] * q[3])))
        xp = float(scene.plate_x()[0])
        ok = bool((scene.depth()[0].abs() < 0.003).all()) and bool(scene.intact()[0].all())
        reads.append((float(kp[0]), float(kp[1]), yaw, xp, ok))
    arr = np.array([r[:4] for r in reads])
    print(f"[smoke] randomization readback (console_x, console_y, console_yaw, "
          f"plate_x):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    sides = {x > 0 for x in arr[:, 3]}
    check("randomization: console xy + yaw and the plate's corridor end / start gap vary "
          "(readback), both ends seen, start always in band, plungers assembled",
          spread[0] > 0.05 and spread[1] > 0.05 and spread[2] > 1.0
          and len(sides) == 2 and all(0.204 <= abs(r[3]) <= 0.220 for r in reads)
          and all(r[4] for r in reads))

    # =========================== 3. null policy =============================================
    env.reset(seed=31)
    x0 = float(scene.plate_x()[0])
    step(600)
    report("null-policy")
    check("null policy: score ~0, plate parked, no success after 600 idle steps",
          float(scene.score()[0]) <= 0.02 and abs(float(scene.plate_x()[0]) - x0) < 0.005
          and not bool(scene.success()[0]))

    # =========================== 4. seed strategy: press the three buttons ==================
    env.reset(seed=41)
    step(60)
    peaks = [press_full(i) for i in (0, 1, 2)]
    report("pressed-all")
    step(840)  # ~7 s: every un-chased button creeps back up
    report("popped")
    d = scene.depth()[0]
    check("seed strategy: all three buttons pressed to the stop (probe is real: 20+ mm "
          "each), shutter untouched -> all pop back up, press-band score only, no success",
          all(p >= 0.020 for p in peaks) and bool((d < 0.005).all())
          and 0.04 <= float(scene.score()[0]) <= 0.06
          and not bool(scene._trapped[0].any()) and not bool(scene.success()[0]))

    # =========================== 5. held press banks nothing ================================
    env.reset(seed=51)
    step(60)
    i_near = 2 if side() > 0 else 0
    peak = press_full(i_near, hold=240)  # hold ~2 s at the bottom
    d_held = float(scene.depth()[0, i_near])
    trapped_held = bool(scene._trapped[0].any())
    ok_held = not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.06
    push_off()
    step(840)  # the spring return creeps at ~4 mm/s: a full 24 mm pop needs ~6-7 s
    report("held-released")
    check("held press: 2 s at the hard stop (no plate chase) banks nothing — no trap, "
          "press-band score, no success; pops on release",
          peak >= 0.020 and d_held >= 0.020 and not trapped_held and ok_held
          and float(scene.depth()[0, i_near]) < 0.005 and not bool(scene.success()[0]))

    # =========================== 6. shove at a raised head ==================================
    env.reset(seed=61)
    step(60)
    i_near = 2 if side() > 0 else 0
    moved, x_end, v_end = shove_at_head()
    rel_z = float(scene._rel(scene.plate)[0, 2])
    d_near = float(scene.depth()[0, i_near])
    report("shoved")
    push_off()
    step(60)
    check("shove at raised head: 6 N ram advances the plate to the first head and stalls "
          "(probe is real: moved, then held still under force), no climb, head not "
          "pressed, nothing banked",
          moved >= 0.003 and 0.193 <= x_end <= 0.206 and v_end < 0.03
          and rel_z < 0.005 and d_near < 0.006
          and float(scene.score()[0]) <= 0.005 and not bool(scene._trapped[0].any()))

    # =========================== 7. lift-out blocked by the lips ============================
    peak_z = -1.0
    for _ in range(180):
        scene.push_w["plate"][0] = scene.up_w()[0] * 12.0
        step(1)
        peak_z = max(peak_z, float(scene._rel(scene.plate)[0, 2]))
    push_off()
    step(90)
    report("lift-tried")
    check("lift-out: a 12 N vertical pull (3x plate weight) rises into the roof lips and "
          "stops (probe is real: took up the 2 mm clearance), plate stays in the corridor",
          0.0008 <= peak_z <= 0.005 and abs(float(scene._rel(scene.plate)[0, 2])) < 0.003
          and float(scene._rel(scene.plate)[0, 1]).__abs__() < c.plate_y_tol)

    # =========================== 8. late chase ==============================================
    env.reset(seed=71)
    step(60)
    i_near = 2 if side() > 0 else 0
    peak = press_full(i_near)
    step(840)  # ~7 s: the button pops fully before the plate moves
    moved, x_end, _v = shove_at_head()
    report("late-chase")
    push_off()
    step(60)
    check("late chase: press-release, wait ~7 s, then push — the popped head blocks the "
          "plate; only the 0.05 press latch remains, no trap, no success",
          peak >= 0.020 and moved >= 0.003 and 0.193 <= x_end <= 0.206
          and float(scene.depth()[0, i_near]) < 0.006
          and not bool(scene._trapped[0].any())
          and 0.04 <= float(scene.score()[0]) <= 0.06 and not bool(scene.success()[0]))

    # =========================== 9. partial coverage (aperture tolerance) ===================
    env.reset(seed=81)
    step(60)
    s = side()
    for i, x in enumerate(G.X_OFF):
        place(scene.plungers[i], (x, 0.0, -0.0125))
    place(scene.plate, (s * 0.037, 0.0, 0.0005))
    step(240)
    report("partial-cover")
    d = scene.depth()[0]
    cov = scene.covered()[0]
    i_far = 0 if s > 0 else 2
    check("partial coverage: plate physically overhangs all three pressed heads but sits "
          "6 mm outside the judged all-covered window — far head physically held yet "
          "judged uncovered, 0.55 band, success stays False",
          bool((d >= 0.008).all()) and bool((d <= 0.016).all())
          and int(cov.sum()) == 2 and not bool(cov[i_far])
          and 0.50 <= float(scene.score()[0]) <= 0.56
          and not bool(scene.success()[0]))

    # =========================== 10. pry-back reverts success (sanctioned) ==================
    allow_success = True
    chase_to(0.0, s)
    step(180)
    got_success = bool(scene.success()[0])
    report("constructed-goal")
    chase_to(s * 0.150, -s, n_max=500)  # pry the plate back toward its end
    allow_success = False
    step(240)
    report("pried-back")
    d = scene.depth()[0]
    i_keep = 2 if s > 0 else 0
    popped = [i for i in range(3) if i != i_keep]
    check("pry-back: constructed success accepted, then prying the plate back pops the "
          "uncovered heads and reverts success LIVE; trapped_ever latches keep 0.80",
          got_success and not bool(scene.success()[0])
          and all(float(d[i]) < 0.006 for i in popped)
          and float(d[i_keep]) >= 0.008
          and 0.79 <= float(scene.score()[0]) <= 0.81)

    # =========================== 11./12. global =============================================
    check("no accidental success: success() never fired outside the sanctioned segment",
          not saw_success)
    fin = all(bool(torch.isfinite(b.data.root_state_w).all())
              for b in (scene.console, scene.plate, *scene.plungers))
    check("final: every body state finite", fin)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shutter_trap")
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

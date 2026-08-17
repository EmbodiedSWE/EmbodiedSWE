"""Smoke / rubric-REJECTION battery for BatonStowScene (sim_gen task
`libero_kitchen_scene5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_i199`)
— NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — slide-force open, wrench-servo threading, slide-
force shut — is the acceptance evidence that the rubric ACCEPTS a correct outcome).
Every teleport here is instrumentation that CONSTRUCTS a wrong (or partial) outcome
as a settled state and asserts the rubric REJECTS it; no probe in this battery ever
reaches success(), and a final audit check asserts exactly that.

  1-2. settle/no-NaN      — reset layout settles finite: drawer shut, both batons
                            resting on the roof, all still, score ~0;
  3-4. randomization      — READBACK over 8 seeded resets: the black/white roof-slot
                            side flips (Bernoulli), per-slot xy jitter and per-baton
                            yaw are real;
  5.  null policy         — 240 idle steps -> score ~0, no success;
  6-7. seed strategy      — the seed task's whole plan (open the drawer, DROP the
                            black item in flat) is geometrically impossible here:
                            the black baton dropped flat over the fully-open mouth —
                            axis-aligned AND at the aperture diagonal — ends NOT
                            inside (the 20 cm baton cannot lie in the 15.7 cm
                            aperture), NOT success;
  8.  leaning near-miss   — baton left tilted, low end on the drawer floor, upper
                            part on the rim: an endpoint is out of the cavity box ->
                            NOT inside, NOT success;
  9.  inside-but-open     — baton constructed lying flat inside the cavity with the
                            drawer fully OPEN -> inside holds but NOT success (the
                            shut clause is load-bearing), score <= 0.85;
  10. latched credit      — teleporting that baton back OUT leaves the latched score
                            unchanged (credit does not evaporate), still no success;
  11. decoy in, shut      — the WHITE decoy sealed inside the shut drawer (black
                            out): the decoy cap crushes the score to <= ~0.10 even
                            though open/thread/inside credit was already latched;
                            NOT success;
  12. both batons in      — black correctly enclosed AND shut AND still, but the
                            decoy also inside -> NOT success, score <= ~0.10 (the
                            "shove both in" strategy is rejected);
  13. crosswise interlock — the black baton laid ACROSS the open drawer on its rims
                            (endpoints never register inside), then a REAL bounded
                            closing force: the drawer verifiably moves but the
                            episode cannot reach success (roof-edge interlock /
                            expulsion — either way rejected);
  14. rejection audit     — success() was never True at ANY judged point;
  15. final no-NaN        — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.<task>.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version and silently rejects RTX -> the
# annotator returns EMPTY frames. Disable the driver check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import threading

import numpy as np
import torch

import robobench
from robobench.core import ENVS

robobench.discover()
try:
    from . import scene as scene_mod  # noqa: F401  (registers)
except ImportError:  # pragma: no cover - forge fallback
    import scene as scene_mod  # noqa: F401

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.baton_stow")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -1.00, 0.85)) + o),
                                tuple(np.array((0.42, 0.00, 0.18)) + o),
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

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action)
            if (annot is not None and step_i % args.record_every == 0
                    and len(frames) < args.max_frames):
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    ever_success = [False]

    def judge() -> tuple[float, bool]:
        s, ok = float(scene.score()[0]), bool(scene.success()[0])
        ever_success[0] = ever_success[0] or ok
        return s, ok

    def open_mm() -> float:
        return float(scene.drawer_open()[0]) * 1000.0

    def body_xyz(body) -> tuple[float, float, float]:
        p = (body.data.root_pos_w - scene.env_origins)[0]
        return float(p[0]), float(p[1]), float(p[2])

    def body_yaw_deg(body) -> float:
        q = body.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def ep_loc() -> list:
        """Drawer-local (x, z) of the two black endpoints."""
        loc = (scene.black_endpoints() - scene.drawer.data.root_pos_w.unsqueeze(1))[0]
        return [(float(loc[i, 0]), float(loc[i, 2])) for i in range(2)]

    def report(tag: str) -> None:
        b = body_xyz(scene.black)
        s, ok = judge()
        print(f"[smoke] {tag:16s} | open={open_mm():5.1f}mm "
              f"black=({b[0]:+.3f},{b[1]:+.3f},{b[2]:.3f}) "
              f"inside={bool(scene.black_inside()[0])} "
              f"decoy_in={bool(scene.decoy_in_cavity()[0])} "
              f"open_max={float(scene._open_max[0]):.2f} thread={bool(scene._thread[0])} "
              f"ins_latch={bool(scene._inside[0])} close_max={float(scene._close_max[0]):.2f} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def place(body, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0], st[:, 1], st[:, 2] = x, y, z
        st[:, 3], st[:, 4], st[:, 5], st[:, 6] = quat
        st[:, 0:3] += scene.env_origins
        body.write_root_state_to_sim(st, all_ids)

    def pose_drawer(open_m: float) -> None:
        """Teleport the drawer along its own prismatic slide coordinate (a joint-arc
        write: the joint stays consistent; the springless slide then holds the pose)."""
        place(scene.drawer, c.front_x - open_m, 0.0, c.drawer_z)

    def q_yaw(deg: float) -> tuple:
        a = math.radians(deg) / 2
        return (math.cos(a), 0.0, 0.0, math.sin(a))

    def q_pitch(deg: float) -> tuple:
        a = math.radians(deg) / 2
        return (math.cos(a), 0.0, math.sin(a), 0.0)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    step(90)
    report("reset-settle")
    fin0 = (torch.isfinite(scene.drawer.data.root_state_w).all()
            and torch.isfinite(scene.black.data.root_state_w).all()
            and torch.isfinite(scene.decoy.data.root_state_w).all())
    _, _, bz = body_xyz(scene.black)
    _, _, dz = body_xyz(scene.decoy)
    still = (float(scene.black.data.root_lin_vel_w[0].norm()) < c.settle_speed
             and float(scene.decoy.data.root_lin_vel_w[0].norm()) < c.settle_speed)
    check("settle: states finite, drawer shut, both batons resting on the roof, all still",
          bool(fin0) and open_mm() < 5.0 and bz > c.roof_z1 - 0.010
          and dz > c.roof_z1 - 0.010 and still)
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(5)
        bx, by, _ = body_xyz(scene.black)
        reads.append((bx, by, body_yaw_deg(scene.black), 1.0 if by > 0 else 0.0))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (black_x, black_y, black_yaw_deg, on_left):\n"
          f"{np.round(arr, 4)}", flush=True)
    flags = arr[:, 3]
    check("randomization: black/white roof-slot side flips across seeded resets AND "
          "the black baton's yaw varies by > 5 deg (readback)",
          0.0 < flags.mean() < 1.0 and (arr[:, 2].max() - arr[:, 2].min()) > 5.0)
    jit = 0.0
    for flag in (0.0, 1.0):
        grp = arr[flags == flag]
        if len(grp) >= 2:
            jit = max(jit, float((grp[:, 0:2].max(axis=0) - grp[:, 0:2].min(axis=0)).max()))
    check("randomization: per-slot spawn jitter is real (readback spread > 4 mm within "
          "a slot group)", jit > 0.004)

    # =========================== 5. null policy fails =======================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.02 and not ok)

    # =========================== 6-7. seed strategy: flat drop is impossible ================
    # The seed's whole plan — open the drawer and DROP the black item in flat — cannot
    # work here: the 20 cm baton does not fit the 15.7 x 12 cm exposed aperture at any
    # yaw. Drop it flat over the fully-open mouth, axis-aligned and at the diagonal.
    for sd, yaw, tag in ((41, 0.0, "flat-drop-axis"), (51, 30.0, "flat-drop-diag")):
        torch.manual_seed(sd)
        env.reset()
        step(30)
        pose_drawer(c.travel - 0.008)
        step(20)
        place(scene.black, 0.345, 0.0, 0.26, quat=q_yaw(yaw))
        step(300)
        report(tag)
        s, ok = judge()
        eps = ep_loc()
        check(f"seed strategy ({tag}): black baton dropped flat over the fully-open "
              "mouth settles NOT inside (an endpoint is out of the cavity box), "
              "NOT success",
              not bool(scene.black_inside()[0]) and not ok and s <= 0.85
              and open_mm() > 140.0)
        print(f"[smoke]   endpoints drawer-local (x, z): {np.round(eps, 3)}", flush=True)

    # =========================== 8. leaning near-miss =======================================
    # A threading attempt abandoned halfway: low end on the drawer floor, upper part
    # resting on the rim — an endpoint stays out of the cavity box.
    torch.manual_seed(61)
    env.reset()
    step(30)
    pose_drawer(c.travel - 0.008)
    step(20)
    place(scene.black, 0.356, 0.0, 0.2122, quat=q_pitch(20.0))  # tip (0.45, 0, 0.178)
    step(240)
    report("leaning")
    s, ok = judge()
    check("leaning near-miss: baton left tilted (low end on the drawer floor, upper "
          "part on the rim) — NOT inside, NOT success, score <= 0.85",
          not bool(scene.black_inside()[0]) and not ok and s <= 0.85)

    # =========================== 9. inside-but-open =========================================
    # Construct the baton lying flat INSIDE the cavity of the fully-open drawer: the
    # inside clause holds live, but success needs the drawer SHUT.
    torch.manual_seed(71)
    env.reset()
    step(30)
    pose_drawer(c.travel - 0.008)
    step(20)
    dx = float((scene.drawer.data.root_pos_w - scene.env_origins)[0, 0])
    place(scene.black, dx + 0.128, 0.0, c.drawer_z + c.drawer_t + 0.018)
    step(120)
    report("inside-open")
    s9, ok = judge()
    check("inside-but-open: baton lying flat inside the OPEN drawer — inside holds "
          "live but NOT success (shut clause load-bearing), 0.69 <= score <= 0.85",
          bool(scene.black_inside()[0]) and open_mm() > 140.0 and not ok
          and 0.69 <= s9 <= 0.85)

    # =========================== 10. latched credit survives regression =====================
    place(scene.black, 0.10, 0.45, c.baton_cross / 2 + 0.002)
    step(60)
    report("regressed")
    s10, ok = judge()
    check("latched credit: teleporting the enclosed baton back OUT to the floor "
          f"leaves the latched score unchanged ({s9:.3f} -> {s10:.3f}), still no "
          "success", abs(s10 - s9) < 1e-3 and not bool(scene.black_inside()[0]) and not ok)

    # =========================== 11. decoy sealed in the shut drawer ========================
    # Same episode (open/thread/inside credit latched at ~0.70): shut the drawer with
    # the WHITE decoy inside. The decoy cap crushes the score.
    pose_drawer(0.002)
    step(20)
    dx = float((scene.drawer.data.root_pos_w - scene.env_origins)[0, 0])
    place(scene.decoy, dx + 0.128, 0.0, c.drawer_z + c.drawer_t + 0.018)
    step(120)
    report("decoy-in-shut")
    s11, ok = judge()
    check("decoy sealed in: WHITE decoy inside the SHUT drawer (black out) — the "
          f"decoy cap crushes the latched {s10:.2f} score to <= 0.105, NOT success",
          bool(scene.decoy_in_cavity()[0]) and open_mm() < 8.0 and not ok
          and s11 <= 0.105)

    # =========================== 12. both batons in =========================================
    # Black correctly enclosed AND shut AND still — but the decoy is inside too.
    place(scene.decoy, dx + 0.128, -0.032, c.drawer_z + c.drawer_t + 0.018)
    place(scene.black, dx + 0.128, 0.032, c.drawer_z + c.drawer_t + 0.018)
    step(150)
    report("both-in-shut")
    s12, ok = judge()
    check("both batons in: black enclosed in the SHUT drawer and still, but the decoy "
          "inside too — NOT success, score <= 0.105 (the 'shove both in' strategy is "
          "rejected)",
          bool(scene.black_inside()[0]) and bool(scene.decoy_in_cavity()[0])
          and open_mm() < 8.0 and not ok and s12 <= 0.105)

    # =========================== 13. crosswise interlock (REAL closing force) ===============
    # Lay the black baton ACROSS the open drawer on its side rims (it never registers
    # inside: endpoints sit above the cavity box), then push the drawer shut with the
    # real bounded slide force. The roof-edge interlock either stalls the drawer or
    # expels the baton — no path to success, and the force verifiably acts.
    torch.manual_seed(81)
    env.reset()
    step(30)
    pose_drawer(c.travel - 0.008)
    step(20)
    place(scene.black, 0.32, 0.0, 0.240, quat=q_yaw(90.0))
    step(60)
    open_start = open_mm()
    min_open = open_start
    for i in range(700):
        x = float(scene.drawer_open()[0])
        v = float(scene.drawer.data.root_lin_vel_w[0, 0])
        f = -60.0 * (0.002 - x) + 8.0 * (-v)  # real push toward shut, same as solve
        scene.drawer_drive[:] = max(-10.0, min(10.0, f))
        env.step(no_action)  # raw env.step here is fine; frame recording is handled by step()
        min_open = min(min_open, open_mm())
        if i % 50 == 0:
            judge()
    scene.drawer_drive[:] = 0.0
    step(150)
    report("crosswise-close")
    s13, ok = judge()
    check("crosswise interlock: baton across the rims never registers inside "
          "(no inside latch), the REAL closing push verifiably moved the drawer "
          f"({open_start:.0f} -> min {min_open:.0f} mm) yet the episode never "
          "reached success",
          not bool(scene._inside[0]) and (open_start - min_open) > 30.0
          and not ok and s13 <= 0.85)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    fin = (torch.isfinite(scene.drawer.data.root_state_w).all()
           and torch.isfinite(scene.black.data.root_state_w).all()
           and torch.isfinite(scene.decoy.data.root_state_w).all()
           and torch.isfinite(scene.shell.data.root_state_w).all())
    check("final: all task-object states finite (no NaN)", bool(fin))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.baton_stow")
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
    except BaseException:  # noqa: BLE001 - Kit threads would hang the interpreter
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

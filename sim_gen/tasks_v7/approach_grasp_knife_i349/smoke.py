"""smoke — REJECTION battery for the TimerFlipDockScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py — one transport hop to an inverted hover +
wrench-servo descent — already proves the rubric ACCEPTS the correct outcome). Every
check here CONSTRUCTS a wrong strategy or a near-miss (teleports are instrumentation)
and asserts the rubric REJECTS it. success() is audited at every judged point and must
NEVER be True anywhere:

  1./2.  settle/no-NaN   — reset settles finite; capsule physically standing collar-down
                           on the sampled ring (radius readback vs sp_r), marble at its
                           body-frame chamber home; score 0, no latches, no success;
  3.     randomization   — READBACK across 8 seeds: dock xy/yaw, ring radius, capsule
                           world xy + spawn yaw all vary; capsule on the ring and the
                           marble at home EVERY seed;
  4.     null policy     — 240 idle steps -> score 0, nothing latched;
  5.     seed strategy   — the seed's whole plan (grasp-lift-CARRY): the capsule is
                           pinned UPRIGHT at carry poses, including the exact hover
                           directly over the well (altitude + over-well verified) ->
                           only `lifted` latches, score exactly 0.10 — carrying the
                           object the seed's way is nearly worthless;
  6.     wrong-end       — PHYSICAL one-way key: dropped UPRIGHT (collar-end first) over
                           the well, the 62 mm collar catches on the rim and the capsule
                           perches 60 mm too high (perch_z readback); tip_in/depth never
                           latch (they gate on inversion), score stays 0.10;
  7.     45 deg yaw      — PHYSICAL: dropped inverted but turned 45 deg, the 57 mm tube
                           diagonal jams on the 50 mm well mouth: the foot tip never
                           crosses the rim plane (min readback), tip_in/depth stay 0;
                           the marble still crosses (honest flip) -> score caps at 0.40;
  8.     wrong place     — right pose, wrong place: standing inverted ON the rim band
                           46 mm off-axis latches nothing new -> 0.40, no success;
  9.     no marble       — positive control of the payload gate: with the marble
                           teleported away the capsule DOES physically seat (readback:
                           collar on rim, centred, inverted) yet success stays False and
                           the score caps at 0.70 — the flip's physical proof is
                           load-bearing;
 10.     extraction      — pulling the seated capsule back out keeps the latched 0.70
                           and success False (no credit evaporates, none is faked);
 11./12. proud hold      — held (pinned) 30 mm short of the seat: partial depth credit
                           only (0.55 + 0.30*depth readback), no success; carrying it
                           away keeps exactly that latched score;
 13./14. audit           — success() never True at any judged point, max score <= 0.77;
                           final no-NaN; frames.npz saved.

Run (forge): python -u -m simgen_tasks.approach_grasp_knife_i349.smoke --headless
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--record_every", type=int, default=8)
parser.add_argument("--max_frames", type=int, default=500)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the L20/4090 driver version and silently rejects RTX -> the
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
    from simgen_tasks.approach_grasp_knife_i349 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.timer_flip_dock")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)

    # --- recording (viewport rgb annotator, the proven forge mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.42, -0.95, 0.75)) + o),
                                tuple(np.array((0.42, 0.0, 0.10)) + o),
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

    never_success = [True]
    max_score = [0.0]

    def judge() -> tuple[float, bool]:
        """Sample the rubric at a judged point and feed the global audit."""
        s = float(scene.score()[0])
        ok = bool(scene.success()[0])
        never_success[0] &= not ok
        max_score[0] = max(max_score[0], s)
        return s, ok

    def timer_local() -> torch.Tensor:
        return scene.world_to_local(scene.timer.data.root_pos_w)[0]

    def foot_local_z() -> float:
        foot, _top = scene.end_tips()
        return float(scene.world_to_local(foot)[0, 2])

    def foot_local_xy() -> tuple[float, float]:
        foot, _top = scene.end_tips()
        fl = scene.world_to_local(foot)[0]
        return float(fl[0]), float(fl[1])

    def up_z() -> float:
        return float(scene.timer_up()[0, 2])

    def report(tag: str) -> None:
        p = timer_local()
        ml = scene.marble_local()[0]
        s, ok = judge()
        print(f"[smoke] {tag:18s} timer=({float(p[0]):+.3f},{float(p[1]):+.3f},"
              f"{float(p[2]):+.3f}) up_z={up_z():+.2f} foot_z={foot_local_z():+.3f} "
              f"marble_l=({float(ml[0]):+.3f},{float(ml[1]):+.3f},{float(ml[2]):+.3f}) "
              f"lift={bool(scene.lifted[0])} inv={bool(scene.inverted[0])} "
              f"tip={bool(scene.tip_in[0])} depth={float(scene.depth_max[0]):.2f} "
              f"mb={bool(scene.marble_across[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def all_finite() -> bool:
        return (bool(torch.isfinite(scene.timer.data.root_state_w).all())
                and bool(torch.isfinite(scene.marble.data.root_state_w).all()))

    # ----- pose constructors (dock-local; instrumentation only) -----------------------------
    def up_state(x: float, y: float, z: float) -> torch.Tensor:
        """13-state UPRIGHT (collar end down), tube faces square to the well."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.local_to_world(torch.tensor([[x, y, z]], device=device), ids)
        half = float(scene.d_yaw[0]) / 2
        st[0, 3], st[0, 6] = math.cos(half), math.sin(half)
        return st

    def inv_state(x: float, y: float, z: float, extra_yaw: float = 0.0) -> torch.Tensor:
        """13-state INVERTED (foot end down): q = qz(d_yaw + extra) * qx(pi)."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = scene.local_to_world(torch.tensor([[x, y, z]], device=device), ids)
        half = (float(scene.d_yaw[0]) + extra_yaw) / 2
        st[0, 4], st[0, 5] = math.cos(half), math.sin(half)
        return st

    def write_assembly(st: torch.Tensor, ml: torch.Tensor) -> None:
        """Coherent transport hop: capsule to `st`, marble re-written at its unchanged
        BODY-frame offset `ml` (the hop never moves the marble relative to the capsule)."""
        scene.timer.write_root_state_to_sim(st, ids)
        sm = torch.zeros(1, 13, device=device)
        sm[0, 0:3] = st[0, 0:3] + quat_apply(st[0:1, 3:7], ml.unsqueeze(0))[0]
        sm[0, 3] = 1.0
        scene.marble.write_root_state_to_sim(sm, ids)

    def pin_timer(st: torch.Tensor, n: int) -> None:
        """Hold the capsule at a fixed pose for n substeps (carry/hold emulation); the
        marble stays FREE inside — gravity does whatever the pose implies."""
        for _ in range(n):
            scene.timer.write_root_state_to_sim(st, ids)
            step(1)

    HOVER_Z = c.rim_z + c.half_len + 0.015  # 0.210: foot tip 15 mm above the rim

    # =========================== 1./2. settle / no-NaN ======================================
    env.reset(seed=11)
    step(90)
    report("reset")
    tl = timer_local()
    ml = scene.marble_local()[0]
    ring_r = math.hypot(float(tl[0]), float(tl[1]))
    check("settle: finite state, capsule standing collar-down ON the sampled ring "
          "(radius readback), marble at its body-frame chamber home, everything still",
          all_finite() and up_z() > 0.99
          and abs(float(tl[2]) - c.spawn_z) < 0.005
          and abs(ring_r - float(scene.sp_r[0])) < 0.005
          and abs(float(ml[2]) - c.marble_home_z) < 0.005
          and bool(scene.settled()[0]))
    check("settle: score 0, no latches, no success",
          float(scene.score()[0]) <= 1e-6 and not bool(scene.lifted[0])
          and not bool(scene.inverted[0]) and not bool(scene.tip_in[0])
          and not bool(scene.marble_across[0]) and float(scene.depth_max[0]) <= 1e-6
          and not bool(scene.success()[0]))

    # =========================== 3. randomization (readback) ================================
    reads, phys_ok = [], []
    for s in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=s)
        step(5)
        tl = timer_local()
        ml = scene.marble_local()[0]
        tw = scene.timer.data.root_pos_w[0] - scene.env_origins[0]
        reads.append((float(scene.d_pos[0, 0]), float(scene.d_pos[0, 1]),
                      float(scene.d_yaw[0]), float(scene.sp_r[0]),
                      float(tw[0]), float(tw[1]), float(scene.t_yaw0[0])))
        phys_ok.append(abs(math.hypot(float(tl[0]), float(tl[1]))
                           - float(scene.sp_r[0])) < 0.006
                       and abs(float(ml[2]) - c.marble_home_z) < 0.006)
    arr = np.array(reads)
    spread = arr.max(axis=0) - arr.min(axis=0)
    print(f"[smoke] randomization readback (dx, dy, dyaw, sp_r, tx, ty, tyaw0):\n{arr}",
          flush=True)
    print(f"[smoke] spreads={spread} on_ring_marble_home={phys_ok}", flush=True)
    check("randomization: dock pose, ring radius, capsule world xy and spawn yaw all "
          "vary across seeds (readback); capsule on the ring + marble at home every seed",
          spread[0] > 0.015 and spread[1] > 0.015 and spread[2] > 0.15
          and spread[3] > 0.02 and spread[4] > 0.05 and spread[5] > 0.05
          and spread[6] > 1.0 and all(phys_ok))

    # =========================== 4. null policy =============================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: 240 idle steps -> score 0, nothing latched, no success",
          float(scene.score()[0]) <= 1e-6 and not bool(scene.lifted[0])
          and not bool(scene.tip_in[0]) and not bool(scene.marble_across[0])
          and not bool(scene.success()[0]))

    # =========================== 5. seed strategy: grasp-lift-carry UPRIGHT =================
    # The seed's whole plan: grasp the object, lift it, hold it. The capsule is pinned
    # UPRIGHT at carry poses — including the EXACT hover directly over the well — with
    # its altitude verified, so the rejection cannot be vacuous. Only `lifted` may latch:
    # carrying the capsule the seed's way parks the run at 0.10 forever.
    env.reset(seed=41)
    step(30)
    alt_ok, over_ok = True, False
    for (px, py, pz) in ((0.20, -0.10, 0.25), (-0.15, 0.15, 0.20), (0.0, 0.0, 0.28)):
        st = up_state(px, py, pz)
        write_assembly(st, scene.marble_local()[0].clone())
        pin_timer(st, 40)
        foot, top = scene.end_tips()
        min_end = float(torch.minimum(foot[0, 2], top[0, 2])) - float(scene.env_origins[0, 2])
        alt_ok &= min_end > c.lift_z + 0.01
        p = timer_local()
        if abs(float(p[0])) < 0.005 and abs(float(p[1])) < 0.005:
            over_ok = True
    report("carry-upright")
    check("seed strategy: carrying the capsule UPRIGHT (incl. hovering right over the "
          "well, altitude verified) latches only `lifted` -> score exactly 0.10",
          alt_ok and over_ok and bool(scene.lifted[0]) and not bool(scene.inverted[0])
          and not bool(scene.tip_in[0]) and not bool(scene.marble_across[0])
          and float(scene.depth_max[0]) <= 1e-6
          and abs(float(scene.score()[0]) - 0.10) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 6. wrong end: the collar is a one-way key ==================
    # PHYSICAL: dropped UPRIGHT (collar-end first) over the well, the 62 mm collar cannot
    # enter the 50 mm opening — the capsule perches with the collar on the rim, 60 mm
    # above the seat. The insertion latches gate on inversion and never fire.
    env.reset(seed=51)
    step(30)
    write_assembly(up_state(0.0, 0.0, c.perch_z + 0.010), scene.marble_local()[0].clone())
    step(240)
    report("wrong-end-drop")
    tl = timer_local()
    check("wrong end: dropped collar-first the capsule perches ON the rim 60 mm above "
          "the seat (perch_z readback), tip_in/depth never latch, score stays 0.10",
          up_z() > 0.90 and abs(float(tl[2]) - c.perch_z) < 0.008
          and not bool(scene.tip_in[0]) and float(scene.depth_max[0]) <= 1e-6
          and not bool(scene.inverted[0]) and not bool(scene.marble_across[0])
          and abs(float(scene.score()[0]) - 0.10) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 7. 45 deg yaw: the diagonal jams ===========================
    # PHYSICAL: dropped inverted but turned 45 deg, the 57 mm tube diagonal cannot pass
    # the 50 mm mouth: the corners catch on the rim and the foot tip never crosses the
    # rim plane (min readback over the whole drop). The marble still falls through the
    # waist (the flip itself is honest) so the run caps at 0.40 with zero insertion.
    env.reset(seed=61)
    step(30)
    write_assembly(inv_state(0.0, 0.0, c.rim_z + 0.005 + c.half_len, extra_yaw=math.pi / 4),
                   scene.marble_local()[0].clone())
    min_fz = float("inf")
    for _ in range(240):
        step(1)
        min_fz = min(min_fz, foot_local_z())
    report("yaw45-jam")
    check("45 deg yaw: the well mouth physically refuses the tube diagonal (foot tip "
          "never below the rim plane, min readback), tip_in/depth stay 0; the honest "
          "flip still pays marble_across -> score caps at 0.40",
          min_fz > c.tip_z_gate - 0.001 and not bool(scene.tip_in[0])
          and float(scene.depth_max[0]) <= 1e-6 and bool(scene.marble_across[0])
          and abs(float(scene.score()[0]) - 0.40) < 5e-3
          and not bool(scene.success()[0]))

    # =========================== 8. right pose, wrong place =================================
    # Standing inverted ON the rim band 46 mm off-axis: the pose is perfect, the place is
    # not — the foot is outside the tip footprint and the centre outside center_tol.
    env.reset(seed=71)
    step(30)
    write_assembly(inv_state(0.046, 0.0, c.rim_z + c.half_len + 0.002),
                   scene.marble_local()[0].clone())
    step(240)
    report("rim-stand")
    tl = timer_local()
    fx, fy = foot_local_xy()
    check("wrong place: standing inverted ON the rim band 46 mm off-axis (readback: on "
          "the band, foot outside the tip footprint) earns 0.40 and never succeeds",
          up_z() < -0.95 and abs(float(tl[2]) - (c.rim_z + c.half_len)) < 0.006
          and math.hypot(fx, fy) > c.tip_xy + 0.01
          and not bool(scene.tip_in[0]) and float(scene.depth_max[0]) <= 1e-6
          and abs(float(scene.score()[0]) - 0.40) < 5e-3
          and not bool(scene.success()[0]))

    # =========================== 9./10. no marble: the payload gate is load-bearing =========
    # Positive control: with the marble teleported far away, the naked capsule dropped
    # inverted + aligned DOES physically seat (collar on the rim, centred — readback
    # proves the mechanics of seating are real) yet success stays False and the score
    # caps at 0.70: the captive payload's crossing is a load-bearing part of the goal,
    # so faking the capsule pose without an honest flip+settle can never win.
    env.reset(seed=81)
    step(30)
    sm = torch.zeros(1, 13, device=device)
    sm[0, 0:3] = scene.local_to_world(
        torch.tensor([[0.60, 0.60, c.marble_r + 0.001]], device=device), ids)
    sm[0, 3] = 1.0
    scene.marble.write_root_state_to_sim(sm, ids)
    step(5)
    scene.timer.write_root_state_to_sim(inv_state(0.0, 0.0, c.seat_z + 0.020), ids)
    step(240)
    report("seated-no-marble")
    tl = timer_local()
    seated_ok = (up_z() < -0.95 and abs(float(tl[0])) <= c.center_tol
                 and abs(float(tl[1])) <= c.center_tol
                 and abs(float(tl[2]) - c.seat_z) <= c.seat_tol)
    check("no marble: the capsule physically seats (readback: inverted, centred, collar "
          "on the rim) but WITHOUT the marble crossing success stays False, score 0.70",
          seated_ok and bool(scene.tip_in[0]) and float(scene.depth_max[0]) > 0.95
          and not bool(scene.marble_across[0])
          and abs(float(scene.score()[0]) - 0.70) < 0.012
          and not bool(scene.success()[0]))
    s_seat = float(scene.score()[0])
    scene.timer.write_root_state_to_sim(inv_state(0.30, -0.30, c.spawn_z), ids)
    step(90)
    report("extracted")
    check("extraction: pulling the seated capsule back out keeps the latched score and "
          "success False (credit neither evaporates nor is faked)",
          abs(float(scene.score()[0]) - s_seat) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 11./12. proud hold: partial depth only =====================
    # The capsule is held (pinned) inverted + aligned over the well, the marble crosses
    # honestly, then it is pin-lowered into the well but STOPPED 30 mm short of the seat
    # and held: only partial depth credit (0.55 + 0.30 * latched depth), no success.
    # Carrying it away (a coherent hop out — never released in the well, where gravity
    # would legitimately finish the job) keeps exactly the latched score.
    env.reset(seed=91)
    step(30)
    st = inv_state(0.0, 0.0, HOVER_Z)
    write_assembly(st, scene.marble_local()[0].clone())
    pin_timer(st, 240)  # marble falls through the waist while held
    mb_ok = bool(scene.marble_across[0])
    z = HOVER_Z
    while z > c.seat_z + 0.030:
        z -= 0.0006
        scene.timer.write_root_state_to_sim(inv_state(0.0, 0.0, z), ids)
        step(1)
    st_hold = inv_state(0.0, 0.0, c.seat_z + 0.030)
    pin_timer(st_hold, 60)
    report("proud-hold")
    d_read = float(scene.depth_max[0])
    s_hold = float(scene.score()[0])
    tl = timer_local()
    check("proud hold: stopped 30 mm short of the seat (readback) the run earns only "
          "0.55 + 0.30*depth partial credit, depth well below 1, no success",
          mb_ok and bool(scene.tip_in[0]) and 0.50 < d_read < 0.80
          and abs(float(tl[2]) - (c.seat_z + 0.030)) < 0.004
          and abs(s_hold - (0.55 + 0.30 * d_read)) < 2e-3
          and not bool(scene.success()[0]))
    write_assembly(inv_state(0.28, 0.28, c.spawn_z), scene.marble_local()[0].clone())
    step(120)
    report("carried-away")
    check("latched credit: carrying the capsule away keeps exactly the proud-hold score, "
          "success stays False",
          abs(float(scene.score()[0]) - s_hold) < 1e-3
          and not bool(scene.success()[0]))

    # =========================== 13./14. audit + final no-NaN ===============================
    check("audit: success() was never True at any judged point; max score <= 0.77",
          never_success[0] and max_score[0] <= 0.77)
    check("final: all body states finite", all_finite())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.timer_flip_dock")
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
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

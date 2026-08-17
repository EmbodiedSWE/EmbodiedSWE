"""Smoke / rubric-REJECTION battery for SpoonKnifeEdgeScene (sim_gen task
`track_spoon_i265`) — NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — drop the cube into the pocket, compute the composed
balance point, perch the loaded spoon on the crest, watch it settle level — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport here is
instrumentation that CONSTRUCTS a wrong (or partial) outcome and asserts the rubric
REJECTS it; no probe in this battery ever reaches success(), and a final audit check
asserts exactly that.

  1.  settle/no-NaN       — reset state settles: spoon flat on the floor, cube on the
                            floor, stand upright; score ~0, no success;
  2.  authored masses     — get_masses() readback: spoon and cube match the
                            cfg-derived masses (the custom spoon spawner must author
                            MassAPI itself; cfg mass_props are ignored there), stand
                            reads its 6 kg;
  3.  randomization       — READBACK over 8 seeded resets: stand xy + yaw, spoon slot
                            xy + free yaw, and cube slot xy all vary;
  4.  null policy         — 240 idle steps -> score ~0, no success;
  5.  SEED strategy       — the seed's whole plan ("carry the spoon and set it down at
                            the goal") = the spoon transported and SET DOWN flat on
                            the slab beside the fin: no load, no balance -> score ~0,
                            NOT success;
  6.  settle gate         — the empty spoon released 15 mm above the crest: mid-fall
                            the state is refused (velocity readback: settled() False,
                            success False at that judged instant);
  7.  empty-spoon balance — ...it then settles LEVEL on the crest at its OWN CoM
                            (positive control: the crest genuinely supports a
                            correctly-placed body, level/perch readback True) — but
                            the cube is not in the pocket: score ~0, NOT success.
                            (Note the empty balance point is ~28 mm from the loaded
                            one — memorizing it does not solve the task.);
  8.  load only           — the cube dropped into the pocket of the floor-lying
                            spoon: real containment, `loaded` latches its 0.20 — and
                            nothing more, NOT success;
  9.  latched credit      — the cube teleported back OUT of the pocket to the floor:
                            the latched 0.20 survives unchanged, still NOT success;
  10. naive midpoint      — the loaded assembly perched with the spoon's GEOMETRIC
                            CENTRE over the crest (the composed balance point is
                            ~48 mm away): it TIPS — level/air readback fails, score
                            stops at the transient latches (<= 0.55), NOT success;
  11. near-miss window    — the composed balance point placed 8 mm OUTSIDE the 10 mm
                            support window: it TIPS the other way — the window is a
                            real physical boundary, NOT success;
  12. wrong place         — the cube resting on the HANDLE (not in the pocket) with
                            the assembly perched at its TRUE composed balance point:
                            it settles LEVEL (readback — equilibrium alone is not the
                            task) but the pocket clause rejects: score ~0, NOT
                            success;
  13. rejection audit     — success() was never True at ANY judged point;
  14. final no-NaN        — all task-object states finite; frames.npz saved.

Run (forge): python -u -m simgen_tasks.track_spoon_i265.smoke --headless
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
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.spoon_knife_edge")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.65)) + o),
                                tuple(np.array((0.40, 0.00, 0.08)) + o),
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

    def deg(x: float) -> float:
        return math.degrees(x)

    def report(tag: str) -> None:
        s, ok = judge()
        print(f"[smoke] {tag:16s} | tilt={deg(float(scene.spoon_tilt()[0])):+6.2f}deg "
              f"pocket={bool(scene.in_pocket()[0])} perched={bool(scene.perched()[0])} "
              f"level={bool(scene.level()[0])} roll={bool(scene.roll_ok()[0])} "
              f"air={bool(scene.ends_air()[0])} settled={bool(scene.settled()[0])} "
              f"loaded_l={bool(scene._loaded[0])} perched_l={bool(scene._perched[0])} "
              f"score={s:.3f} success={ok} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def write_state(body, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = pos_w
        st[:, 3:7] = quat_w
        body.write_root_state_to_sim(st, all_ids)

    def place(body, wx: float, wy: float, z: float,
              quat=(1.0, 0.0, 0.0, 0.0), settle_steps: int = 30) -> None:
        """Kinematic probe placement (instrumentation, not a solution) + REAL physics
        steps before judging (the zero-step trap)."""
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0], pos[:, 1], pos[:, 2] = wx, wy, z
        pos += scene.env_origins
        q = torch.tensor(quat, device=device).expand(n, 4).clone()
        write_state(body, pos, q)
        step(settle_steps)

    def drop_into_pocket(settle_steps: int = 90) -> None:
        """Hover the cube 25 mm above the floor-lying spoon's open pocket (yaw
        aligned) and let gravity seat it — the same honest load as solve P1."""
        sp_pos = scene.spoon.data.root_pos_w.clone()
        sp_q = scene.spoon.data.root_quat_w.clone()
        hov = torch.tensor([c.bowl_cx, 0.0, c.floor_t + c.rim_h + 0.025 + c.cube_s / 2],
                           device=device).expand(n, 3)
        write_state(scene.cube, sp_pos + quat_apply(sp_q, hov), sp_q)
        step(settle_steps)
        for _ in range(10):
            if float(scene.cube.data.root_lin_vel_w.norm(dim=-1)[0]) < 0.02:
                break
            step(30)

    def perch(origin_x_local: float, cube_rel="keep", hover: float = 0.003) -> None:
        """Teleport the spoon to a hover above the crest with its origin at the given
        stand-local x, axis across the fin, then release. cube_rel: "keep" carries
        the cube at its live relative pose (assembly transport), "skip" leaves the
        cube where it is, a tuple places it at that explicit spoon-frame pose."""
        st_pos = scene.stand.data.root_pos_w.clone()
        st_q = scene.stand.data.root_quat_w.clone()
        tgt = torch.tensor([origin_x_local, 0.0, c.crest_top + hover],
                           device=device).expand(n, 3)
        spoon_pos_t = st_pos + quat_apply(st_q, tgt)
        b_rel = cube_q_t = None
        if cube_rel == "keep":
            b_rel = scene.cube_in_spoon()[0]
            q_rel = quat_mul(quat_inv(scene.spoon.data.root_quat_w),
                             scene.cube.data.root_quat_w)
            cube_q_t = quat_mul(st_q, q_rel)
        elif cube_rel != "skip":
            b_rel = torch.tensor(cube_rel, device=device)
            cube_q_t = st_q.clone()
        write_state(scene.spoon, spoon_pos_t, st_q.clone())
        if b_rel is not None:
            write_state(scene.cube,
                        spoon_pos_t + quat_apply(st_q, b_rel.unsqueeze(0).expand(n, 3)),
                        cube_q_t)

    def settle_perch(max_blocks: int = 30) -> None:
        # break on the scene's own rest gate (velocity thresholds + pose-stillness
        # window); RAW angular velocity keeps a persistent GPU phantom readback
        # (~0.2 rad/s) on the crest contact even with the pose frozen sub-mm
        for _ in range(max_blocks):
            step(30)
            if bool(scene.settled()[0]):
                break

    def composed_bal() -> float:
        b_rel = scene.cube_in_spoon()[0]
        return (c.m_spoon * c.com_x + c.m_cube * float(b_rel[0])) \
            / (c.m_spoon + c.m_cube)

    def spoon_z() -> float:
        return float((scene.spoon.data.root_pos_w - scene.env_origins)[0][2])

    def fin_all() -> bool:
        ok = True
        for b in (scene.stand, scene.spoon, scene.cube):
            ok = ok and bool(torch.isfinite(b.data.root_state_w).all())
        return ok

    def yaw_of(q: torch.Tensor) -> float:
        return 2.0 * math.atan2(float(q[3]), float(q[0]))

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(240)
    report("reset-settled")
    s, ok = judge()
    check("settle: states finite; spoon flat on the floor (origin z <= 12 mm), cube "
          "on the floor, stand upright, everything settled; score ~0, no success",
          fin_all() and spoon_z() <= 0.012 and bool(scene.stand_upright()[0])
          and bool(scene.settled()[0]) and s <= 0.03 and not ok)

    # =========================== 2. authored masses readback ================================
    sm = float(scene.spoon.root_physx_view.get_masses().flatten()[0])
    km = float(scene.cube.root_physx_view.get_masses().flatten()[0])
    tm = float(scene.stand.root_physx_view.get_masses().flatten()[0])
    print(f"[smoke] mass readback: spoon={sm * 1000:.2f}g (cfg {c.m_spoon * 1000:.2f}g) "
          f"cube={km * 1000:.2f}g (cfg {c.m_cube * 1000:.2f}g) stand={tm:.2f}kg", flush=True)
    check("authored masses: spoon and cube read back the cfg-derived masses (the "
          "custom spoon spawner authors MassAPI mass+CoM itself) and the stand reads "
          "its 6 kg",
          abs(sm - c.m_spoon) < 1e-3 and abs(km - c.m_cube) < 1e-3
          and abs(tm - c.stand_mass) < 0.05)

    # =========================== 3. randomization is real ===================================
    reads = []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        env.reset(seed=sd)
        step(5)
        sp = (scene.stand.data.root_pos_w - scene.env_origins)[0]
        pp = (scene.spoon.data.root_pos_w - scene.env_origins)[0]
        kp = (scene.cube.data.root_pos_w - scene.env_origins)[0]
        reads.append((float(sp[0]), float(sp[1]), yaw_of(scene.stand.data.root_quat_w[0]),
                      float(pp[0]), float(pp[1]), yaw_of(scene.spoon.data.root_quat_w[0]),
                      float(kp[0]), float(kp[1])))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (stand_x, stand_y, stand_yaw, spoon_x, "
          f"spoon_y, spoon_yaw, cube_x, cube_y):\n{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: stand xy + yaw, spoon slot xy + free yaw, and cube slot xy "
          "all vary across seeded resets (readback)",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > 0.05
          and spread[3] > 0.01 and spread[4] > 0.01 and spread[5] > 0.5
          and spread[6] > 0.01 and spread[7] > 0.01)

    # =========================== 4. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: score ~0 and no success after 240 idle steps", s <= 0.03 and not ok)

    # =========================== 5. SEED strategy ===========================================
    # The seed's whole plan is "carry the (already-grasped) spoon along a path and set
    # it down at the goal". Here that end state — the EMPTY spoon transported and SET
    # DOWN flat on the stand's slab, right beside the fin — must be worthless.
    env.reset(seed=41)
    step(120)
    st_pos = scene.stand.data.root_pos_w.clone()
    st_q = scene.stand.data.root_quat_w[0]
    syaw = yaw_of(st_q)
    lx, ly = 0.055, 0.0  # stand-local: on the slab, clear of the fin
    wx = float(st_pos[0, 0] - scene.env_origins[0, 0]) + math.cos(syaw) * lx - math.sin(syaw) * ly
    wy = float(st_pos[0, 1] - scene.env_origins[0, 1]) + math.sin(syaw) * lx + math.cos(syaw) * ly
    qy = syaw + math.pi / 2  # spoon axis ALONG the fin direction, lying on the slab
    place(scene.spoon, wx, wy, c.slab_t + 0.004,
          (math.cos(qy / 2), 0.0, 0.0, math.sin(qy / 2)), settle_steps=120)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy (the spoon carried and SET DOWN flat on the slab beside the "
          "fin): no load, no balance — score <= 0.03, NOT success",
          spoon_z() > 0.012 and s <= 0.03 and not ok)

    # =========================== 6-7. settle gate + empty-spoon balance =====================
    # The EMPTY spoon released 15 mm above the crest at its OWN CoM point: mid-fall
    # the judged state must be refused (settle gate); it then settles LEVEL — the
    # crest genuinely supports a correctly-placed body — but the pocket clause alone
    # rejects success (level equilibrium is not the task).
    env.reset(seed=51)
    step(120)
    perch(-c.com_x, cube_rel="skip", hover=0.015)  # the cube stays on the floor
    step(3)  # mid-fall
    settled_mid = bool(scene.settled()[0])
    v_mid = float(scene.spoon.data.root_lin_vel_w.norm(dim=-1)[0])
    s_mid, ok_mid = judge()
    check("settle gate: mid-fall the spoon reads unsettled (|v| readback > "
          f"settle_lin) and success is False at that judged instant",
          not settled_mid and v_mid > c.settle_lin and not ok_mid)
    settle_perch()
    report("empty-balance")
    s, ok = judge()
    check("empty-spoon balance (positive control): the EMPTY spoon placed at its own "
          "CoM settles LEVEL on the crest (level + perch geometry readback True) — "
          "but no cube in the pocket: score <= 0.03, NOT success. (The empty balance "
          "point sits ~28 mm from the loaded one — memorizing it cannot solve the "
          "task.)",
          bool(scene.level()[0]) and bool(scene.perched()[0])
          and bool(scene.ends_air()[0]) and not bool(scene.in_pocket()[0])
          and s <= 0.03 and not ok)

    # =========================== 8-9. load only + latched credit ============================
    env.reset(seed=61)
    step(240)
    drop_into_pocket()
    report("load-only")
    s_load, ok = judge()
    check("load only: the cube dropped into the pocket of the floor-lying spoon — "
          "real containment (in_pocket readback), `loaded` latches its 0.20 and "
          "nothing more; the spoon never perched: NOT success",
          bool(scene.in_pocket()[0]) and spoon_z() <= 0.012
          and 0.18 <= s_load <= 0.25 and not ok)
    place(scene.cube, 0.90, -0.50, c.cube_s / 2 + 0.002, settle_steps=60)
    report("cube-removed")
    s_out, ok = judge()
    check("latched credit: teleporting the cube back OUT of the pocket leaves the "
          "latched 0.20 unchanged (and still no success)",
          not bool(scene.in_pocket()[0]) and abs(s_out - s_load) < 0.02
          and s_out >= 0.18 and not ok)

    # =========================== 10. naive midpoint perch tips ==============================
    env.reset(seed=71)
    step(240)
    drop_into_pocket()
    bal = composed_bal()
    perch(0.0)  # spoon MIDPOINT over the crest — the balance point is ~48 mm away
    settle_perch()
    report("naive-midpoint")
    tilt = deg(float(scene.spoon_tilt()[0]))
    s, ok = judge()
    check("naive midpoint perch: supporting the spoon at its geometric centre "
          f"(composed balance point {bal * 1000:+.1f} mm away) TIPS it — the settled "
          "state fails level/air/perch, score stops at the transient latches "
          "(<= 0.55), NOT success",
          not (bool(scene.level()[0]) and bool(scene.ends_air()[0])
               and bool(scene.perched()[0]))
          and s <= 0.55 and not ok)

    # =========================== 11. near-miss just OUTSIDE the window ======================
    env.reset(seed=81)
    step(240)
    drop_into_pocket()
    bal = composed_bal()
    off = c.crest_half + 0.008  # 8 mm past the crest edge — outside the support window
    perch(-bal + off)
    settle_perch()
    report("near-miss")
    tilt = deg(float(scene.spoon_tilt()[0]))
    s, ok = judge()
    check("near-miss: the composed balance point placed 8 mm OUTSIDE the 10 mm "
          f"support window tips the spoon (settled tilt {tilt:+.1f} deg or ends out "
          "of air) — the window is a real physical boundary, NOT success",
          not (bool(scene.level()[0]) and bool(scene.ends_air()[0])
               and bool(scene.perched()[0]))
          and s <= 0.55 and not ok)

    # =========================== 12. balances but the cube is in the WRONG place ============
    env.reset(seed=91)
    step(240)
    cube_x = 0.030  # on the HANDLE, spoon frame
    bal12 = (c.m_spoon * c.com_x + c.m_cube * cube_x) / (c.m_spoon + c.m_cube)
    perch(-bal12, cube_rel=(cube_x, 0.0, c.handle_t + c.cube_s / 2 + 0.002))
    settle_perch()
    report("wrong-place")
    s, ok = judge()
    b_rel = scene.cube_in_spoon()[0]
    check("wrong place: the cube resting on the HANDLE with the assembly perched at "
          "its true composed balance point settles LEVEL (readback — equilibrium "
          "alone is not the task) but in_pocket is False: score <= 0.03, NOT success",
          bool(scene.level()[0]) and bool(scene.perched()[0])
          and not bool(scene.in_pocket()[0]) and float(b_rel[2]) < 0.05
          and s <= 0.03 and not ok)

    # =========================== 13-14. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", fin_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.spoon_knife_edge")
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
    main()

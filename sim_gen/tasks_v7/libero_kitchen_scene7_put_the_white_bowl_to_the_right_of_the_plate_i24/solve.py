"""Teleport solution for MugHookScene (sim_gen task
libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i24) — the
task's legitimacy certificate.

The load-bearing interactions and how each is executed:

1. TRANSPORT (teleport, small per-step pose increments): the mug is "held" by
   writing its root pose each physics step along a smooth path — lift off the
   floor, carry to a PRE-STAGING point on the BLUE peg's axis ~110 mm off the tip,
   then slide down the axis to the STAGING pose: the handle aperture aligned with
   the peg axis, its center 10 mm BEYOND the tip — deliberately NOT threaded
   (the scene's axis segment stops 25 mm short of the tip, so the staged loop is
   ~35 mm from the segment, outside the 30 mm gate; asserted). Per-step increments
   are <= ~2 mm / <= ~1.5 deg. Which peg is blue is READ BACK per episode (the
   color->slot permutation is random), never assumed.
2. THREADING (forces + contact, the core interaction): positional teleports stop.
   The "wrist" keeps holding the mug's ORIENTATION (each step the orientation is
   re-written to the staging frame while the POSITION is copied back unchanged
   from the physics readback — a rigid wrist on a compliant arm), and all
   TRANSLATION is driven by `set_external_force_and_torque` at the CoM: gravity
   support (m*g up), a gentle push along the peg axis toward the panel, and a
   small radial centering term, under a velocity governor. The loop slides over
   the peg by force and contact; the `thread` latch can only set during this
   phase (staging kept it outside the gate; asserted). If the loop misses
   (radial escape), the attempt is re-staged and retried.
3. HANG (gravity + contact, the goal state): all external forces are cleared —
   hands off. The mug falls a few millimetres until the peg catches the top bar
   of the handle loop, slides down the 12-deg-tilted peg toward the panel, swings,
   and settles DANGLING: the peg carries the whole weight through the loop,
   nothing under the mug but air. Every clause success() checks (threaded,
   airborne, settled) is produced by gravity and contact — an unsupported mug at
   that pose would simply fall.

Prints `SIM_GEN_SCORE <score>` at each phase boundary (non-decreasing: the
scene's partial credit is latched), then holds HANDS-OFF for >= 3.3 simulated
seconds after success() first turns True and prints `SIM_GEN_SOLVE: SUCCESS`
only if success() still holds.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_to_the_right_of_the_plate_i24.solve --headless [--seed N]
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SOLVE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

_ = scene_mod  # imported for its registrations / helpers


def _slerp(q0: torch.Tensor, q1: torch.Tensor, f: float) -> torch.Tensor:
    """Batched quaternion slerp (wxyz), shortest arc."""
    d = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(d < 0, -q1, q1)
    d = d.abs().clamp(max=1.0)
    th = torch.acos(d)
    s = torch.sin(th).clamp(min=1e-6)
    q = (torch.sin((1 - f) * th) / s) * q0 + (torch.sin(f * th) / s) * q1
    q = torch.where(th < 1e-3, (1 - f) * q0 + f * q1, q)
    return q / q.norm(dim=-1, keepdim=True)


def main() -> None:
    from isaaclab.utils.math import quat_apply, quat_from_matrix

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.mug_hook")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)

    # Seed AFTER build (the EnvCfg.build reseed trap); print the description so the
    # task statement is on record.
    env.reset(seed=args.seed)
    print("[solve] " + "=" * 70, flush=True)
    print(env.scene.describe(), flush=True)
    print("[solve] " + "=" * 70, flush=True)

    def step(k: int) -> None:
        for _ in range(k):
            env.step(no_action)

    def report(tag: str) -> None:
        mz = float((scene.mug.data.root_pos_w - scene.env_origins)[0, 2])
        d_b = float(scene.dist_to_peg("blue")[0])
        d_r = float(scene.dist_to_peg("red")[0])
        d_g = float(scene.dist_to_peg("green")[0])
        v = float(scene.mug.data.root_lin_vel_w.norm(dim=-1)[0])
        print(f"[solve] {tag:14s} | mug_z={mz:.3f} d_blue={d_b * 1000:5.1f}mm "
              f"d_red={d_r * 1000:5.1f}mm d_green={d_g * 1000:5.1f}mm v={v:.3f} "
              f"lift={bool(scene._lift[0])} thread={bool(scene._thread[0])} "
              f"threaded_now={bool(scene.threaded()[0])} "
              f"airborne={bool(scene.airborne()[0])} settled={bool(scene.mug_settled()[0])} "
              f"success={bool(scene.success()[0])} score={float(scene.score()[0]):.3f}",
              flush=True)

    def print_score(tag: str) -> float:
        s = float(scene.score()[0])
        print(f"[solve] phase boundary: {tag}", flush=True)
        print(f"SIM_GEN_SCORE {s:.4f}", flush=True)
        return s

    def hold_pose(p_w: torch.Tensor, q: torch.Tensor) -> None:
        """One held-pose write: mug root at world `p_w`, orientation `q`, zero vel."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = p_w
        st[:, 3:7] = q
        scene.mug.write_root_state_to_sim(st, all_ids)

    def glide(p0: torch.Tensor, p1: torch.Tensor, q0: torch.Tensor,
              q1: torch.Tensor, steps: int) -> None:
        """Carry the held mug smoothly p0->p1 / q0->q1 over `steps` physics steps
        (per-step increments stay small — the poses a gripper would impose)."""
        for s_i in range(1, steps + 1):
            f = s_i / steps
            hold_pose(p0 + (p1 - p0) * f, _slerp(q0, q1, f))
            env.step(no_action)

    zero1 = torch.zeros(n, 1, 3, device=device)

    def clear_forces() -> None:
        scene.mug.set_external_force_and_torque(zero1, zero1, env_ids=all_ids)

    # ---------------- phase 0: reset, settle, baseline -------------------------------------
    step(180)  # everything comes to rest on the floor
    slots = scene.peg_slot[0].tolist()
    root, dirw = scene.peg_frame("blue")
    root = root.clone()
    dirw = dirw.clone()  # kinematic pegs: frozen for the episode
    mug0 = (scene.mug.data.root_pos_w - scene.env_origins)[0]
    print(f"[solve] layout readback (seed {args.seed}): peg slots (r,g,b)={slots} "
          f"blue_root=({float(root[0, 0]):+.3f},{float(root[0, 1]):+.3f},"
          f"{float(root[0, 2]):+.3f}) "
          f"blue_dir=({float(dirw[0, 0]):+.2f},{float(dirw[0, 1]):+.2f},"
          f"{float(dirw[0, 2]):+.2f}) "
          f"mug=({float(mug0[0]):+.3f},{float(mug0[1]):+.3f},{float(mug0[2]):+.3f})",
          flush=True)
    report("reset")
    assert torch.isfinite(scene.mug.data.root_pos_w).all(), "NaN/inf after settle"
    assert float(mug0[2]) < 0.05, "mug must start on the floor"
    s0 = print_score("P0 reset+settle (mug on the floor)")
    assert not bool(scene.success()[0]), "fresh reset must not be success"
    assert s0 <= 0.03, f"baseline score should be ~0, got {s0}"

    # ---------------- staging geometry (from readback, per episode) -------------------------
    that = -dirw                                     # threading direction (into the panel)
    up = torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3)
    z_ax = up - (up * that).sum(-1, keepdim=True) * that
    z_ax = z_ax / z_ax.norm(dim=-1, keepdim=True)    # "mug upright" axis, perp to the peg
    x_ax = torch.cross(that, z_ax, dim=-1)           # right-handed [x, y=that, z]
    rot = torch.stack([x_ax, that, z_ax], dim=-1)    # columns = mug local axes in world
    q_stage = quat_from_matrix(rot)
    tip = root + c.peg_len * dirw
    ap_stage = tip + 0.010 * dirw                    # aperture center: 10 mm OFF the tip
    ap_off_w = quat_apply(q_stage, scene._ap_local.expand(n, 3))
    p_stage = ap_stage - ap_off_w                    # mug origin at staging
    p_pre = p_stage + 0.10 * dirw                    # pre-staging: 110 mm off the tip

    # ---------------- phase 1: TRANSPORT — lift, carry, align off the tip -------------------
    p0 = scene.mug.data.root_pos_w.clone()
    q0 = scene.mug.data.root_quat_w.clone()
    p_up = p0.clone()
    p_up[:, 2] = p_pre[:, 2]
    glide(p0, p_up, q0, q_stage, 130)      # lift straight up, rotating to the staging frame
    glide(p_up, p_pre, q_stage, q_stage, 150)  # carry to the pre-staging point on the axis
    glide(p_pre, p_stage, q_stage, q_stage, 80)  # slide down the axis, stop OFF the tip
    report("staged")
    d_stage = float(scene.dist_to_peg("blue")[0])
    assert bool(scene._lift[0]), "lift latch did not set during the carry"
    assert d_stage > c.thread_gate, \
        f"staging must be OUTSIDE the thread gate (got {d_stage * 1000:.1f}mm)"
    assert not bool(scene._thread[0]), "thread latch must not set from teleported staging"
    s1 = print_score("P1 mug staged off the blue peg tip (aligned, NOT threaded)")
    assert s1 >= s0 - 1e-6 and abs(s1 - c.w_lift) < 0.02, f"P1 score {s1} (expect lift only)"

    # ---------------- phase 2: THREADING — force-driven, contact-guided ---------------------
    mg = c.mug_mass * 9.81
    push = 0.40 * mg
    ez_w = up

    def axis_readout() -> tuple[float, float, torch.Tensor]:
        rel = scene.aperture_w() - root
        s_ax = (rel * dirw).sum(-1)
        r_vec = rel - s_ax.unsqueeze(-1) * dirw
        return float(s_ax[0]), float(r_vec.norm(dim=-1)[0]), r_vec

    def hold_wrist() -> None:
        """Rigid wrist, compliant arm: re-write ORIENTATION only; the position is
        copied back unchanged from the physics readback, linear velocity kept —
        translation stays entirely force/contact-driven."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.mug.data.root_pos_w
        st[:, 3:7] = q_stage
        st[:, 7:10] = scene.mug.data.root_lin_vel_w
        scene.mug.write_root_state_to_sim(st, all_ids)

    deep = False
    for attempt in range(3):
        for i in range(700):
            s_ax, rad, r_vec = axis_readout()
            if s_ax <= 0.045:
                deep = True
                break
            if rad > 0.040:  # the loop escaped sideways — missed the peg
                print(f"[solve] threading radial escape (rad={rad * 1000:.1f}mm)",
                      flush=True)
                break
            hold_wrist()
            v = scene.mug.data.root_lin_vel_w.norm(dim=-1)
            f_corr = -(r_vec / max(rad, 1e-6)) * (0.30 * mg) * min(rad / 0.010, 1.0)
            f_vec = mg * ez_w + f_corr + torch.where(
                (v < 0.12).unsqueeze(-1), push * that, torch.zeros_like(that))
            scene.mug.set_external_force_and_torque(
                f_vec.unsqueeze(1), zero1, env_ids=all_ids, is_global=True)
            env.step(no_action)
            if i % 150 == 0:
                print(f"[solve] thread[{attempt}] i={i} s_ax={s_ax * 1000:.1f}mm "
                      f"rad={rad * 1000:.1f}mm", flush=True)
        clear_forces()
        if deep:
            break
        print(f"[solve] threading attempt {attempt} failed; re-staging", flush=True)
        pc = scene.mug.data.root_pos_w.clone()
        qc = scene.mug.data.root_quat_w.clone()
        glide(pc, p_pre, qc, q_stage, 120)
        glide(p_pre, p_stage, q_stage, q_stage, 80)
    if not deep:
        report("THREAD-STUCK")
        print("SIM_GEN_SOLVE: FAIL (loop never threaded over the blue peg)", flush=True)
        os._exit(1)
    s_ax, rad, _rv = axis_readout()
    print(f"[solve] threaded deep: s_ax={s_ax * 1000:.1f}mm rad={rad * 1000:.1f}mm",
          flush=True)
    report("threaded")
    assert bool(scene._thread[0]), "thread latch did not set during the force push"
    s2 = print_score("P2 handle loop threaded over the blue peg (forces still on record)")
    assert s2 >= s1 - 1e-6 and s2 >= c.w_lift + c.w_thread - 0.01, \
        f"P2 score {s2} (expect lift+thread=0.45)"

    # ---------------- phase 3: HANG — release, gravity carries the mug through the loop -----
    clear_forces()
    step(360)  # 3 s: drop onto the peg, slide toward the panel, swing, settle
    ok_live = bool(scene.success()[0])
    for _ in range(6):  # allow extra swing decay if needed
        if ok_live:
            break
        step(120)
        ok_live = bool(scene.success()[0])
    report("hang")
    if not ok_live:
        report("FAIL-state")
        print("SIM_GEN_SOLVE: FAIL (no success after releasing the mug)", flush=True)
        os._exit(1)
    s3 = print_score("P3 mug released — dangling from the blue peg by its handle")
    assert s3 >= s2 - 1e-6, "score decreased across the release"

    # ---------------- phase 4: persistence (>= 3 simulated seconds, hands-off) --------------
    holds = True
    for _ in range(10):  # 10 x 40 steps = 400 substeps = 3.33 s at 120 Hz
        step(40)
        holds = holds and bool(scene.success()[0])
    report("persist")
    s4 = print_score("P4 persistence 3.3 s")
    ok = holds and bool(scene.success()[0]) and s4 >= s3 - 1e-6
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
    except BaseException as exc:  # noqa: BLE001 - die loudly, don't wait for the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SOLVE: FAIL (exception: {exc!r})", flush=True)
        os._exit(2)

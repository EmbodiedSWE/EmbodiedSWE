"""Smoke / rubric-REJECTION battery for SauceBalanceScene (sim_gen task
`living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i253`) —
NullRobot, teleported probe states, RECORDED.

This is NOT a solution (solve.py — load the can, closed-loop greedy
counterweighting off the tilt readback, the sustained equilibrium — is the
acceptance evidence that the rubric ACCEPTS a correct outcome). Every teleport
here is instrumentation that CONSTRUCTS a wrong (or partial) outcome as a settled
state and asserts the rubric REJECTS it; no probe in this battery ever reaches
success(), and a final audit check asserts exactly that.

  1-2.  settle/no-NaN    — reset settles finite: empty beam LEVEL, can and blocks
                           on the floor, still, score ~0;
  3-4.  randomization    — READBACK over 8 seeded resets: stand xy/yaw and the
                           floor-slot shuffle vary; the can's MASS varies (physx
                           get_masses readback — the hidden weigh-target is real);
  5.   null policy       — 240 idle steps -> score ~0, no success, and the empty
                           beam stays level (a level beam alone earns nothing);
  6.   seed strategy     — the seed's whole plan (put the can in the tray) as a
                           settled end state: can seated in the brown pan, nothing
                           else -> the beam pegs at its stop BY ITSELF; loaded is
                           the 0.20 floor, score <= 0.201, no success;
  7.   wrong direction   — counterweights without a load: blocks stacked in the
                           blue pan, can on the floor -> beam pegs counter-side,
                           score ~0 (all counter/level credit is load-gated);
  8.   can not presented — can lying SIDEWAYS across the tray pan -> upright
                           check rejects, no load credit;
  9.   one granule short — can loaded + the exact subset MINUS 50 g -> settles
                           ~4.8 deg tray-down, outside the 3.5 deg band: near
                           may latch, level never, score <= 0.601, no success;
  10.  latched credit    — the can then yanked back to the floor: latched score
                           holds, still no success;
  11.  one granule over  — fresh episode, exact subset PLUS 50 g -> settles
                           counter-side down, same cap, no success;
  12.  anti-prop tower   — all four blocks stacked on the floor under the sunken
                           tray pan -> the tower reaches no beam underside, the
                           beam stays pegged, no level/near credit;
  13.  held-level fake   — beam + load + WRONG single counterweight teleported
                           to level in one burst, then hands off: the level
                           streak dies in ~19 substeps (equilibrium proof), the
                           beam falls back out of the band, success never fires;
  14.  rejection audit   — success() was never True at ANY judged point;
  15.  final no-NaN      — all task-object states finite at the end.

Run (forge): python -u -m simgen_tasks.living_room_scene3_pick_up_the_tomato_sauce_and_put_it_in_the_tray_i253.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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
    env = ENVS.get("simgen.sauce_balance")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    edges = [scene_mod._block_edge(m, c.block_density) for m in c.block_masses]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.95, -1.25, 0.85)) + o),
                                tuple(np.array((0.00, -0.10, 0.25)) + o),
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

    def tilt() -> float:
        return float(scene.tilt_deg()[0])

    def rel(body) -> torch.Tensor:
        return (body.data.root_pos_w - scene.env_origins)[0]

    def report(tag: str) -> None:
        cp = rel(scene.can)
        s, ok = judge()
        inb = scene.blocks_in_counter()[0]
        print(f"[smoke] {tag:16s} | tilt={tilt():+6.2f} "
              f"can=({float(cp[0]):+.3f},{float(cp[1]):+.3f},{float(cp[2]):.3f}) "
              f"in_tray={bool(scene.can_in_tray()[0])} "
              f"blk_in={[int(b) for b in inb]} "
              f"loaded={bool(scene._loaded[0])} counter={bool(scene._counter[0])} "
              f"near={bool(scene._near[0])} level={bool(scene._level[0])} "
              f"lvl_streak={int(scene._lvl_streak[0])} score={s:.3f} success={ok} "
              f"frames={len(frames)}", flush=True)

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

    def place_on_beam(body, local_xyz, extra_quat=None) -> None:
        """Set-down aligned with the (possibly tilted/yawed) beam frame."""
        p_l = torch.tensor(local_xyz, device=device).expand(n, 3)
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.beam_point_w(p_l)
        q = scene.beam.data.root_quat_w
        if extra_quat is not None:
            q = scene_mod._qmul(q, torch.tensor(extra_quat, device=device).expand(n, 4))
        st[:, 3:7] = q
        body.write_root_state_to_sim(st, all_ids)

    def load_can() -> None:
        place_on_beam(scene.can, (c.pan_x, 0.0, c.well_floor_top + c.can_h / 2 + 0.003))

    def stack_in_counter(idxs) -> None:
        """One write burst: the given blocks stacked bottom-up in the blue pan."""
        z = c.well_floor_top
        for i in idxs:
            place_on_beam(scene.blocks[i], (-c.pan_x, 0.0, z + edges[i] / 2 + 0.003))
            z += edges[i] + 0.001

    def subset_for(target: float) -> list[int]:
        """Exact block subset for a target mass (cfg-asserted to exist)."""
        for k in range(16):
            sub = [i for i in range(4) if k >> i & 1]
            if abs(sum(c.block_masses[i] for i in sub) - target) < 1e-6:
                return sorted(sub, key=lambda i: -c.block_masses[i])
        raise AssertionError(f"no subset for {target}")

    def yaw_deg(body) -> float:
        q = body.data.root_quat_w[0]
        return math.degrees(2.0 * math.atan2(float(q[3]), float(q[0])))

    def finite_all() -> bool:
        ok = (torch.isfinite(scene.stand.data.root_state_w).all()
              and torch.isfinite(scene.beam.data.root_state_w).all()
              and torch.isfinite(scene.can.data.root_state_w).all())
        for b in scene.blocks:
            ok = ok and torch.isfinite(b.data.root_state_w).all()
        return bool(ok)

    # =========================== 1-2. settle / no-NaN =======================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("show")
    cp = rel(scene.can)
    still = (float(scene.can.data.root_lin_vel_w[0].norm()) < 0.10
             and float(scene.beam.data.root_ang_vel_w[0].norm()) < 0.10)
    check("settle: states finite, empty beam LEVEL, can and blocks resting on the "
          f"floor, everything still (tilt={tilt():+.2f})",
          finite_all() and abs(tilt()) < 1.0 and still
          and float(cp[2]) < 0.10 and not bool(scene.can_in_tray()[0])
          and not bool(scene.blocks_in_counter()[0].any()))
    s, ok = judge()
    check("settle: score ~0 at reset (<= 0.02), no success", s <= 0.02 and not ok)

    # =========================== 3-4. randomization is real =================================
    reads, mass_reads = [], []
    for sd in (21, 22, 23, 24, 25, 26, 27, 28):
        torch.manual_seed(sd)
        env.reset()
        step(2)
        sp, cq, b0 = rel(scene.stand), rel(scene.can), rel(scene.blocks[0])
        reads.append([float(sp[0]), float(sp[1]), yaw_deg(scene.stand),
                      float(cq[0]), float(cq[1]), float(b0[0])])
        mass_reads.append(float(scene.can.root_physx_view.get_masses().sum()))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (sx, sy, s_yaw, can_x, can_y, w200_x):\n"
          f"{arr}", flush=True)
    print(f"[smoke] can mass readback (kg): {[f'{m:.3f}' for m in mass_reads]}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    yaws = arr[:, 2] % 360.0
    yaw_spread = yaws.max() - yaws.min()
    check("randomization: stand xy/yaw and the floor-slot shuffle vary across seeded "
          f"resets (stand xy {spread[0]:.3f}/{spread[1]:.3f}, yaw {yaw_spread:.1f} deg, "
          f"can x {spread[3]:.3f}, w200 x {spread[5]:.3f})",
          min(spread[0], spread[1]) > 0.01 and yaw_spread > 25.0
          and spread[3] > 0.05 and spread[5] > 0.05)
    m_spread = max(mass_reads) - min(mass_reads)
    check("randomization: the can's hidden MASS varies across resets (physx "
          f"get_masses readback, spread {m_spread * 1000:.0f} g)", m_spread > 0.04)

    # =========================== 5. null policy =============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    s, ok = judge()
    check("null policy: 240 idle steps — score ~0, no success, and the empty beam "
          f"stays level (tilt={tilt():+.2f}; a level beam alone earns nothing)",
          s <= 0.02 and not ok and abs(tilt()) < 1.0)

    # =========================== 6. seed strategy ===========================================
    # The seed's WHOLE plan — pick the can, put it in the tray — as a settled end
    # state. The beam pegs at its stop by itself: the load is the 0.20 floor.
    torch.manual_seed(41)
    env.reset()
    step(30)
    load_can()
    step(360)
    report("seed-strategy")
    s, ok = judge()
    check("seed strategy end state: can seated in the brown pan, nothing else — the "
          f"beam pegged tray-side by itself (tilt={tilt():+.2f} > {c.near_tol_deg}), "
          f"loaded latch only, score <= 0.201 (s={s:.3f}), no success",
          bool(scene.can_in_tray()[0]) and bool(scene._loaded[0])
          and tilt() > c.near_tol_deg and not ok and s <= 0.201)

    # =========================== 7. counterweight without a load ============================
    torch.manual_seed(46)
    env.reset()
    step(30)
    stack_in_counter([0, 1])
    step(360)
    report("no-load")
    s, ok = judge()
    check("wrong direction: blocks stacked in the blue pan with the can still on the "
          f"floor — beam pegged counter-side (tilt={tilt():+.2f}), all credit is "
          f"load-gated, score ~0 (s={s:.3f})",
          bool(scene.blocks_in_counter()[0].any()) and tilt() < -c.near_tol_deg
          and not ok and s <= 0.02)

    # =========================== 8. can not presented (sideways) ============================
    torch.manual_seed(51)
    env.reset()
    step(30)
    hs = math.sqrt(0.5)
    place_on_beam(scene.can, (c.pan_x, 0.0, c.well_floor_top + c.wall_h + c.can_r + 0.003),
                  extra_quat=(hs, 0.0, hs, 0.0))  # 90 deg about beam y: lying down
    step(360)
    report("sideways-can")
    s, ok = judge()
    check("can not presented: can lying SIDEWAYS across the tray pan — the upright "
          f"check rejects it, no load credit (s={s:.3f}), no success",
          not bool(scene.can_in_tray()[0]) and not bool(scene._loaded[0])
          and not ok and s <= 0.02)

    # =========================== 9. one granule short =======================================
    torch.manual_seed(61)
    env.reset()
    step(30)
    m_can = float(scene._can_mass[0])
    load_can()
    step(300)
    sub = subset_for(m_can - c.can_mass_step) if m_can - c.can_mass_step > 1e-6 else []
    print(f"[smoke] granule-short: can {m_can * 1000:.0f} g, subset "
          f"{[scene.BLOCK_NAMES[i] for i in sub]}", flush=True)
    if sub:
        stack_in_counter(sub)
    step(600)
    report("granule-short")
    s, ok = judge()
    t9 = tilt()
    ok9_geom = (t9 > c.level_tol_deg) if sub else (t9 > c.near_tol_deg)
    check("one granule short: the exact counterweight MINUS 50 g settled in the pan — "
          f"the beam holds {t9:+.2f} deg tray-down, OUTSIDE the level band; level "
          f"never latches, score <= 0.601 (s={s:.3f}), no success",
          bool(scene.can_in_tray()[0]) and ok9_geom
          and not bool(scene._level[0]) and not ok and s <= 0.601)

    # =========================== 10. latched credit survives regression =====================
    s_before, _ = judge()
    place(scene.can, 0.35, 0.30, c.can_h / 2 + 0.003)  # yank the can back to the floor
    step(120)
    report("regressed")
    s_a, ok_a = judge()
    step(60)
    s_b, ok_b = judge()
    check("latched credit: the can yanked back out to the floor — the latched score "
          f"holds ({s_before:.3f} -> {s_a:.3f} -> {s_b:.3f}), still no success",
          abs(s_a - s_before) < 1e-3 and abs(s_b - s_a) < 1e-3
          and not ok_a and not ok_b)

    # =========================== 11. one granule over =======================================
    torch.manual_seed(71)
    env.reset()
    step(30)
    m_can = float(scene._can_mass[0])
    load_can()
    step(300)
    sub = subset_for(m_can + c.can_mass_step)
    print(f"[smoke] granule-over: can {m_can * 1000:.0f} g, subset "
          f"{[scene.BLOCK_NAMES[i] for i in sub]}", flush=True)
    stack_in_counter(sub)
    step(600)
    report("granule-over")
    s, ok = judge()
    t11 = tilt()
    check("one granule over: the exact counterweight PLUS 50 g — the beam settles "
          f"counter-side down ({t11:+.2f} deg), outside the level band; no level "
          f"latch, score <= 0.601 (s={s:.3f}), no success",
          bool(scene.can_in_tray()[0]) and t11 < -c.level_tol_deg
          and not bool(scene._level[0]) and not ok and s <= 0.601)

    # =========================== 12. anti-prop tower ========================================
    torch.manual_seed(81)
    env.reset()
    step(30)
    load_can()
    step(300)
    t_before = tilt()
    # world xy under the sunken tray pan (beam-frame pan centre projected down)
    pan_w = scene.beam_point_w(torch.tensor([c.pan_x, 0.0, c.well_floor_top],
                                            device=device).expand(n, 3))
    px = float((pan_w - scene.env_origins)[0, 0])
    py = float((pan_w - scene.env_origins)[0, 1])
    z = 0.0
    for i in range(4):  # full tower, heaviest at the bottom
        place(scene.blocks[i], px, py, z + edges[i] / 2 + 0.001)
        z += edges[i] + 0.001
    step(300)
    report("prop-tower")
    s, ok = judge()
    check("anti-prop: the full 4-block tower stacked under the sunken tray pan "
          f"reaches no beam underside — the beam stays pegged ({t_before:+.2f} -> "
          f"{tilt():+.2f} deg), no near/level credit, no success",
          tilt() > c.near_tol_deg and not bool(scene._near[0])
          and not bool(scene._level[0]) and not ok and s <= 0.201)

    # =========================== 13. held-level fake ========================================
    # Beam + loaded can + a WRONG single counterweight teleported to LEVEL in one
    # coherent burst (write the whole linkage — a lone member gets depenetrated),
    # then hands off: the streak must die in ~19 substeps and the beam must leave
    # the band — the level streak is an equilibrium proof, not a pose check.
    torch.manual_seed(91)
    env.reset()
    step(30)
    m_can = float(scene._can_mass[0])
    wrong = 3  # w50 alone: imbalance >= 100 g for every can mass — the damped
    # escape from the band is decisively faster than the 75-substep streak
    st = scene.stand.data.root_state_w[all_ids].clone()
    st[:, 7:13] = 0.0
    beam_st = st.clone()
    beam_st[:, 2] += c.hinge_h  # level pose, stand yaw
    scene.stand.write_root_state_to_sim(st, all_ids)
    scene.beam.write_root_state_to_sim(beam_st, all_ids)
    load_can()
    stack_in_counter([wrong])
    max_streak, ever13 = 0, False
    for _ in range(300):
        env.step(no_action)
        max_streak = max(max_streak, int(scene._lvl_streak[0]))
        ever13 = ever13 or bool(scene.success()[0])
    report("held-level-fake")
    s, ok = judge()
    ever_success[0] = ever_success[0] or ever13
    check("held-level fake: beam forced level with the can loaded and a WRONG "
          f"counterweight ({scene.BLOCK_NAMES[wrong]} vs {m_can * 1000:.0f} g can) — "
          f"the level streak dies at {max_streak} < {c.level_streak} substeps, the "
          f"beam falls back out (tilt={tilt():+.2f}), success never fires",
          max_streak < c.level_streak and not ever13 and not ok
          and abs(tilt()) > c.level_tol_deg)

    # =========================== 14-15. audit + no-NaN ======================================
    check("rejection audit: success() was never True at any judged point in this battery",
          not ever_success[0])
    check("final: all task-object states finite (no NaN)", finite_all())

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.sauce_balance")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        for nm, okc in checks:
            if not okc:
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
    try:
        main()
    except BaseException as exc:  # noqa: BLE001 — die loudly, never hang until the watchdog
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(exc).__name__}: {exc})", flush=True)
        os._exit(1)

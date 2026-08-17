"""Smoke battery for CounterBalanceScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — drop the exact binary subset of
counterweights into the empty pan and let the passive plant swing level; the Franka
strategy is TASK.md's embodiment argument). All probes here move bodies by the same
TRANSPORT-teleport used by solve.py (hover above a seat, release with zero velocity);
every judged outcome — every tilt — is produced by the passive D6 + gravity plant.

11 named checks:
  1. settle    — clean reset: finite state everywhere, the beam physically RESTS
                 pinned toward the cargo side (readback ~stop angle), the sampled
                 cargo physically seated in the sampled pan, authored beam mass took
                 (get_masses readback), score ~0, no success;
  2. random    — cargo mass k, cargo side and the weight slot permutation all vary
                 across 8 seeds; cargo seated readback on EVERY draw;
  3. null      — 2.5 s of nothing: beam stays pinned, score < 0.02, no success;
  4. negative  — the SEED's plan (move THE object to a goal pose): carrying the
                 CARGO off the scale relevels the beam PHYSICALLY — and earns
                 nothing (success requires the cargo to stay aboard);
  5. negative  — near miss UNDER: the subset for k-1 units settles the beam still
                 tipped toward the cargo, far outside tol — partial credit only;
  6. negative  — near miss OVER: the subset for k+1 units settles the beam tipped
                 the OTHER way — overshoot fails exactly like undershoot;
  7. negative  — split pans: an extra weight in the CARGO pan plus a heavier subset
                 opposite levels the beam PHYSICALLY — and fails (weights may only
                 ride in the counter pan);
  8. negative  — wrong object: a spare cargo cylinder used as a counterweight
                 levels the beam PHYSICALLY — and fails (only brass counts);
  9. exactness — the exact subset -> success() and score == 1.0, still true 1 s
                 later, hands off;
 10. latch     — plucking one counterweight back off revokes success (success is
                 live state); the latched share of earned credit remains;
 11. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.robosuite_env_i175.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the driver version and
# silently rejects RTX -> the annotator returns EMPTY frames.
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
    from simgen_tasks.robosuite_env_i175 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Binary decompositions over the {1, 2, 4} brass set (solve.py's map, extended to the
# off-by-one probes) and the collision-free pan-local y seats per subset size.
SUBSET = {1: [1], 2: [2], 3: [2, 1], 4: [4], 5: [4, 1], 6: [4, 2], 7: [4, 2, 1]}
Y_SEATS = {1: [0.0], 2: [-0.040, 0.040], 3: [0.0, -0.048, 0.048]}


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.counter_balance")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.tensor([0], device=device)
    tol = math.radians(c.tol_deg)

    from isaaclab.utils.math import quat_apply  # noqa: E402

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.95, -0.95, 1.15)) + o),
                                tuple(np.array((0.0, 0.0, 0.50)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def wait_quiet(budget: int) -> bool:
        for _ in range(budget // 10):
            if bool(scene.quiet()[0]):
                return True
            step(10)
        return bool(scene.quiet()[0])

    def report(tag: str) -> None:
        print(f"[smoke] {tag:16s} | tilt={math.degrees(float(scene.tilt()[0])):+6.2f}deg "
              f"counter={float(scene.counter_mass()[0]) / c.unit_mass:.1f}u "
              f"cargo={int(scene.k[0])}u side={'R' if float(scene.side[0]) > 0 else 'L'} "
              f"quiet={bool(scene.quiet()[0])} A={float(scene.a_latch[0]):.2f} "
              f"B={float(scene.b_latch[0]):.2f} score={float(scene.score()[0]):.3f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def finite_all() -> bool:
        return all(bool(torch.isfinite(b.data.root_state_w).all())
                   for b in scene._bodies().values()) \
            and bool(torch.isfinite(scene.score()).all())

    # --- transport-teleport helpers (identical physics role to solve.py's) ---
    def place(body, pos_local: tuple) -> None:
        st = torch.zeros(1, 13, device=device)
        st[:, 0:3] = scene.env_origins[0:1] + torch.tensor([pos_local], device=device)
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, ids)

    def drop_in_pan(body, s: float, y_local: float, half_h: float) -> None:
        """Hover `body` 10 mm above its seat in pan `s` (live pose readback), release."""
        pan = scene.pans[s]
        local = torch.tensor([[0.0, y_local, c.pan_floor[2] / 2 + half_h + 0.010]],
                             device=device)
        pos = pan.data.root_pos_w[0:1] + quat_apply(pan.data.root_quat_w[0:1], local)
        st = torch.zeros(1, 13, device=device)
        st[:, 0:3] = pos
        st[:, 3] = 1.0
        body.write_root_state_to_sim(st, ids)

    def drop_subset(units_total: int, s: float) -> None:
        """Drop SUBSET[units_total] into pan `s`, largest first, then settle."""
        chosen = SUBSET[units_total]
        seats = Y_SEATS[len(chosen)]
        for i, u in enumerate(chosen):
            drop_in_pan(scene.weights[u], s, seats[i], c.weight_h / 2)
            step(30)
            wait_quiet(1200)

    def park_weight(u: int) -> None:
        """Transport a weight back to a clear spot on the table (off the scale)."""
        i = list(c.weight_units).index(u)
        place(scene.weights[u],
              (c.slot_x[i], c.slot_y - 0.10, c.table_top_z + c.weight_h / 2 + 0.005))

    # --- probe seed: need k in {3, 5} — the masses that admit BOTH a level-but-illegal
    # split-pan configuration over {1,2,4} and a spare cargo of k-1 units ---
    seed_pick = None
    for sd in range(0, 24):
        env.reset(seed=sd)
        step(2)
        if int(scene.k[0]) in (3, 5):
            seed_pick = sd
            break
    assert seed_pick is not None, "no probe seed in 0..23 draws k in {3, 5}"
    kk = int(scene.k[0])
    print(f"[smoke] probe seed={seed_pick} cargo={kk}u "
          f"side={'R' if float(scene.side[0]) > 0 else 'L'}", flush=True)

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=seed_pick)
    ok_q = wait_quiet(1440)
    report("reset")
    side = float(scene.side[0])
    tilt0 = math.degrees(float(scene.tilt()[0]))
    beam_mass = float(scene.beam.root_physx_view.get_masses().reshape(-1)[0])
    print(f"[smoke] beam mass readback = {beam_mass:.3f} kg (authored {c.beam_mass})",
          flush=True)
    check("settle: clean reset — finite, beam RESTS pinned toward the cargo side, cargo "
          "seated, authored mass took, score ~0",
          finite_all() and ok_q and tilt0 * side > 0
          and 9.0 <= abs(tilt0) <= c.tilt_limit_deg + 0.7
          and bool(scene.cargo_seated()[0]) and abs(beam_mass - c.beam_mass) < 0.01
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. randomization across seeds ==================================
    draws, seated_all = [], True
    for sd in (11, 12, 13, 14, 15, 16, 17, 18):
        env.reset(seed=sd)
        step(30)
        seated_all &= bool(scene.cargo_seated()[0])
        slots = []
        for u in c.weight_units:  # which scatter slot each weight landed in (readback)
            x = float((scene.weights[u].data.root_pos_w[0] - scene.env_origins[0])[0])
            slots.append(int(np.argmin([abs(x - s) for s in c.slot_x])))
        draws.append((int(scene.k[0]), "R" if float(scene.side[0]) > 0 else "L",
                      tuple(slots)))
    print(f"[smoke] draws (k, side, weight slots): {draws}", flush=True)
    check("randomization is real (cargo mass k, cargo side, weight scatter all vary; "
          "cargo seated on every draw)",
          len({d[0] for d in draws}) >= 3 and len({d[1] for d in draws}) == 2
          and len({d[2] for d in draws}) >= 3 and seated_all)

    # ========================= 3. null policy =================================================
    env.reset(seed=seed_pick)
    step(300)  # 2.5 s of nothing
    report("null")
    check("null policy: beam stays pinned, score < 0.02, no success",
          abs(math.degrees(float(scene.tilt()[0]))) > 9.0
          and float(scene.score()[0]) < 0.02 and not bool(scene.success()[0]))

    # ========================= 4. negative: the SEED's plan (move THE object) ================
    env.reset(seed=seed_pick)
    wait_quiet(1440)
    place(scene.cargos[kk], (-0.25, 0.30, c.table_top_z + c.cargo_h(kk) / 2 + 0.005))
    step(30)
    wait_quiet(1800)
    report("cargo-removed")
    check("negative (seed strategy): carrying the cargo off the scale relevels the beam "
          "PHYSICALLY — and earns nothing",
          abs(math.degrees(float(scene.tilt()[0]))) < c.tol_deg and finite_all()
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05)

    # ========================= 5. negative: near miss UNDER ===================================
    env.reset(seed=seed_pick)
    wait_quiet(1440)
    drop_subset(kk - 1, -side)
    report("under-by-one")
    t = math.degrees(float(scene.tilt()[0]))
    check("negative (near miss under): one unit short settles the beam still tipped "
          "toward the cargo, outside tol — no success",
          t * side > 0 and 4.5 <= abs(t) <= c.tilt_limit_deg + 0.7
          and not bool(scene.success()[0])
          and 0.20 < float(scene.score()[0]) < 0.55)

    # ========================= 6. negative: near miss OVER ====================================
    env.reset(seed=seed_pick)
    wait_quiet(1440)
    drop_subset(kk + 1, -side)
    report("over-by-one")
    t = math.degrees(float(scene.tilt()[0]))
    check("negative (near miss over): one unit too many tips the beam the OTHER way — "
          "overshoot fails like undershoot",
          t * side < 0 and abs(t) >= 4.5 and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.55)

    # ========================= 7. negative: split pans ========================================
    # An extra weight rides WITH the cargo; a heavier subset opposite. Net torque zero:
    # the beam levels PHYSICALLY — but weights_ok() forbids brass in the cargo pan.
    env.reset(seed=seed_pick)
    wait_quiet(1440)
    u_extra = 1  # k=3: cargo+1 vs [4]; k=5: cargo+1 vs [4,2]
    drop_in_pan(scene.weights[u_extra], side, -0.055, c.weight_h / 2)
    step(30)
    wait_quiet(1200)
    for i, u in enumerate([u for u in SUBSET[kk + u_extra] if u != u_extra]):
        drop_in_pan(scene.weights[u], -side, Y_SEATS[2][i], c.weight_h / 2)
        step(30)
        wait_quiet(1200)
    step(120)
    wait_quiet(1200)
    report("split-pans")
    check("negative (split pans): balancing with a weight riding in the CARGO pan levels "
          "the beam PHYSICALLY — and fails",
          abs(math.degrees(float(scene.tilt()[0]))) < 2.0 * c.tol_deg
          and bool(scene.quiet()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.55)

    # ========================= 8. negative: wrong object as counterweight =====================
    # A spare cargo of k-1 units plus the 1-unit weight sums to k: level — and illegal.
    env.reset(seed=seed_pick)
    wait_quiet(1440)
    drop_in_pan(scene.cargos[kk - 1], -side, -0.045, c.cargo_h(kk - 1) / 2)
    step(30)
    wait_quiet(1200)
    drop_in_pan(scene.weights[1], -side, 0.045, c.weight_h / 2)
    step(30)
    wait_quiet(1500)
    report("wrong-object")
    check("negative (wrong object): a spare cargo cylinder as makeweight levels the beam "
          "PHYSICALLY — and fails (only brass in the counter pan counts)",
          abs(math.degrees(float(scene.tilt()[0]))) < 2.0 * c.tol_deg
          and bool(scene.quiet()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.55)

    # ========================= 9. exactness: success == score 1.0 =============================
    env.reset(seed=seed_pick)
    wait_quiet(1440)
    drop_subset(kk, -side)
    got = False
    for _ in range(120):
        if bool(scene.success()[0]):
            got = True
            break
        step(10)
    report("goal-state")
    good = got and abs(float(scene.score()[0]) - 1.0) < 1e-6
    step(120)  # ...and it persists hands-off
    report("goal-state+1s")
    check("exactness: the exact subset -> success() and score == 1.0, still true 1 s later",
          good and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-6 and finite_all())

    # ========================= 10. achievement latch ==========================================
    u_out = SUBSET[kk][-1]  # pluck the last-dropped weight back off the scale
    park_weight(u_out)
    step(30)
    wait_quiet(1800)
    report("revoked")
    t = math.degrees(float(scene.tilt()[0]))
    check("achievement latch: removing one counterweight revokes success (live state); "
          "latched credit remains",
          not bool(scene.success()[0]) and t * side > 0 and abs(t) > 4.5
          and 0.50 < float(scene.score()[0]) < 0.65)

    # ========================= 11. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.counter_balance")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

"""Smoke battery for LibrarianExtractScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — a gravity-feedforward fingertip
servo that pivots the requested book out on its bottom-front edge, then transports it
to the tray; the Franka strategy is TASK.md's embodiment argument). Wrong outcomes here
are CONSTRUCTED as settled physical states (teleport + settle + readback) and asserted
rejected; the one success construct proves the rubric's ceiling is exactly 1.0.

One linear run, 12 named checks:
  1. settle    — clean reset: finite state everywhere, all five books physically
                 standing AT their sampled slots (readback), score ~0, no success;
  2. readback  — the kinematic layout is physically where the episode tensors say:
                 request card under the TARGET slot, cheeks at row_c +- span/2,
                 tray floor at (tray_x, tray_y) (world-pose vs formula < 2 mm);
  3. random    — 8 seeds: requested book varies (>= 3 distinct), row permutation
                 varies (>= 4 distinct), row_c and tray_y vary; books AT their slots
                 on every draw (readback);
  4. null      — 2 s of nothing: score < 0.05, no success;
  5. negative  — the SEED's plan (put a book ON the shelf, additively): the target
                 laid flat on TOP of the case above its own slot — settled, earns
                 ~0 (the zc gate keeps the tip latch dark), no success;
  6. negative  — orientation near miss: the target STANDING upright in the tray
                 center — not flat, no success, only the latched carry-out credit;
  7. negative  — position near miss: the target flat INSIDE the tray walls but
                 off-center beyond tray_xy_tol (probe seed with the short yellow
                 book, the only one with wall room past the tolerance) — no success;
  8. negative  — wrong object: a NON-requested book laid flat in the tray (target
                 left standing in the row) — score ~0, no success;
  9. negative  — collateral: target correctly flat in the tray BUT one other book
                 toppled out onto the table — the do-not-disturb clause: no success,
                 only the live tray credit (~0.30);
 10. exactness — the goal state (target flat in tray, row intact, settled) ->
                 success() and score == 1.0 exactly, still true 1 s later;
 11. latch     — removing the book from the tray (stood back in its slot) revokes
                 success (live state); only the latched carry-out share remains;
 12. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.put_books_on_bookshelf_i190.smoke --headless
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
    from simgen_tasks.put_books_on_bookshelf_i190 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

QX90 = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)  # R_x(90 deg): thickness axis -> world z
QID = (1.0, 0.0, 0.0, 0.0)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.librarian_extract")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids0 = torch.tensor([0], device=device)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.72, -0.62, 1.05)) + o),
                                tuple(np.array((-0.05, 0.0, 0.48)) + o),
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

    def report(tag: str) -> None:
        print(f"[smoke] {tag:16s} | pitch={math.degrees(float(scene.pitch_fwd()[0])):+6.1f}deg "
              f"tip={float(scene.tip_latch[0]):.2f} out={float(scene.out_latch[0]):.0f} "
              f"others={bool(scene.others_ok()[0])} tray={bool(scene.tray_ok()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def tp(body, x: float, y: float, z: float, quat: tuple = QID) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor((x, y, z), device=device) + scene.env_origins[0]
        st[0, 3:7] = torch.tensor(quat, device=device)
        body.write_root_state_to_sim(st, ids0)

    def finite_all() -> bool:
        bodies = list(scene.books) + list(scene.kin_parts.values())
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies) \
            and bool(torch.isfinite(scene.score()).all())

    def book_pos(b: int) -> torch.Tensor:
        return scene.books[b].data.root_pos_w[0] - scene.env_origins[0]

    def kin_pos(nm: str) -> torch.Tensor:
        return scene.kin_parts[nm].data.root_pos_w[0] - scene.env_origins[0]

    def up_z(b: int) -> float:
        from isaaclab.utils.math import quat_apply

        v = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0)
        return float(quat_apply(scene.books[b].data.root_quat_w[:1], v)[0, 2])

    def ey_z(b: int) -> float:
        from isaaclab.utils.math import quat_apply

        v = torch.tensor([0.0, 1.0, 0.0], device=device).unsqueeze(0)
        return float(quat_apply(scene.books[b].data.root_quat_w[:1], v)[0, 2])

    def books_at_slots(tol: float = 0.006) -> bool:
        ok = True
        for b in range(c.n_books):
            p = book_pos(b)
            ok &= abs(float(p[1]) - float(scene.slot_y[0, b])) < tol
            ok &= abs(float(p[0]) - c.book_x) < tol
            ok &= up_z(b) > 0.99
        return ok

    # --- probe-seed search (deterministic, readback-based, no stepping needed) ---------------
    # A: case-top construct must physically FIT on the ceiling board: the flat book spans
    #    slot_y +- h/2 along y and the board spans row_c +- 0.105.
    # B: the off-center-in-tray near miss needs wall room past tray_xy_tol: only the short
    #    yellow book (index 3, h=0.16) leaves (0.28 - 0.16)/2 = 0.06 > 0.055.
    seed_a = seed_b = None
    for sd in range(0, 21):
        env.reset(seed=sd)
        tgt = int(scene.target_idx[0])
        sy = float(scene.slot_y[0, tgt])
        rc = float(scene.row_c[0])
        if seed_a is None and abs(sy - rc) + c.book_h[tgt] / 2 <= 0.100:
            seed_a = sd
        if seed_b is None and tgt == 3:
            seed_b = sd
        if seed_a is not None and seed_b is not None:
            break
    assert seed_a is not None and seed_b is not None, \
        f"no probe seeds in 0..20 (case-top={seed_a}, yellow-target={seed_b})"
    print(f"[smoke] probe seeds: case-top={seed_a}, yellow-target={seed_b}", flush=True)

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=0)
    step(90)
    report("reset")
    tgt = int(scene.target_idx[0])
    t_t, h_t = c.book_t[tgt], c.book_h[tgt]
    tray_y = float(scene.tray_y[0])
    check("settle: clean reset (finite, books physically AT their slots, score ~0)",
          finite_all() and books_at_slots()
          and abs(math.degrees(float(scene.pitch_fwd()[0]))) < 2.0
          and float(scene.tip_latch[0]) < 0.02 and float(scene.out_latch[0]) == 0.0
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. kinematic layout readback ===================================
    rc = float(scene.row_c[0])
    cheek_y = c.inner_span / 2 + c.cheek_t / 2
    err = 0.0
    for nm, want in (
        ("card", (c.card_x, float(scene.slot_y[0, tgt]), c.z0 + c.card_size[2] / 2)),
        ("case_cheek_l", (c.shelf_x, rc - cheek_y, (c.zs + c.zc) / 2)),
        ("case_cheek_r", (c.shelf_x, rc + cheek_y, (c.zs + c.zc) / 2)),
        ("tray_floor", (c.tray_x, tray_y, c.z0 + c.tray_floor_size[2] / 2)),
    ):
        got = kin_pos(nm)
        err = max(err, float((got - torch.tensor(want, device=device)).norm()))
    print(f"[smoke] layout readback: max err={err * 1000:.2f}mm "
          f"(card under slot of book{tgt})", flush=True)
    check("readback: card under the TARGET slot, cheeks and tray at the sampled poses (< 2 mm)",
          err < 0.002)

    # ========================= 3. randomization across seeds ==================================
    targets, perms, rcs, trays = [], [], [], []
    ok_slots = True
    for sd in range(2, 10):
        env.reset(seed=sd)
        step(30)
        targets.append(int(scene.target_idx[0]))
        perms.append(tuple(int(i) for i in torch.argsort(scene.slot_y[0]).tolist()))
        rcs.append(round(float(scene.row_c[0]), 4))
        trays.append(round(float(scene.tray_y[0]), 4))
        ok_slots &= books_at_slots()
    print(f"[smoke] draws: targets={targets} perms={perms}", flush=True)
    print(f"[smoke]        row_c={rcs} tray_y={trays}", flush=True)
    check("randomization is real (target, permutation, row_c, tray_y all vary; readback holds)",
          len(set(targets)) >= 3 and len(set(perms)) >= 4
          and len(set(rcs)) >= 5 and max(rcs) - min(rcs) > 0.015
          and len(set(trays)) >= 5 and max(trays) - min(trays) > 0.04
          and ok_slots)

    # ========================= 4. null policy =================================================
    env.reset(seed=0)
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. negative: the SEED's plan (put a book ON the shelf) =========
    env.reset(seed=seed_a)
    step(60)
    tgt_a = int(scene.target_idx[0])
    t_a, h_a = c.book_t[tgt_a], c.book_h[tgt_a]
    sy_a = float(scene.slot_y[0, tgt_a])
    tp(scene.books[tgt_a], c.shelf_x, sy_a, c.zc + c.ceil_t + t_a / 2 + 0.004, QX90)
    step(90)
    report("case-top")
    pz = float(book_pos(tgt_a)[2])
    check("negative (seed strategy): book laid ON TOP of the case — settled there, earns ~0, "
          "no success",
          abs(pz - (c.zc + c.ceil_t + t_a / 2)) < 0.006 and abs(ey_z(tgt_a)) > 0.96
          and float(scene.tip_latch[0]) < 0.02 and float(scene.out_latch[0]) == 0.0
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]) and finite_all())

    # ========================= 6. negative: STANDING in the tray (orientation near miss) ======
    env.reset(seed=0)
    step(60)
    tp(scene.books[tgt], c.tray_x, tray_y, c.tray_floor_top + h_t / 2 + 0.0005, QID)
    step(90)
    report("standing-in-tray")
    check("negative (near miss, orientation): target STANDING upright in the tray — no "
          "success, only latched carry-out credit",
          up_z(tgt) > 0.9 and not bool(scene.tray_ok()[0])
          and not bool(scene.success()[0])
          and 0.10 < float(scene.score()[0]) < 0.40)

    # ========================= 7. negative: off-center in the tray (position near miss) =======
    env.reset(seed=seed_b)
    step(60)
    ty_b = float(scene.tray_y[0])
    t_b, h_b = c.book_t[3], c.book_h[3]  # yellow: h=0.16 — 0.06 wall room > 0.055 tol
    off = c.tray_xy_tol + 0.003  # 58 mm off-center: outside tol, 2 mm clear of the wall
    tp(scene.books[3], c.tray_x, ty_b + off, c.tray_floor_top + t_b / 2 + 0.001, QX90)
    step(90)
    report("off-center")
    p_b = book_pos(3)
    inside_wall = abs(float(p_b[1]) - ty_b) + h_b / 2 < c.tray_floor_size[1] / 2 - c.tray_wall_t
    check("negative (near miss, position): flat INSIDE the tray but beyond tray_xy_tol — "
          "no success",
          abs(ey_z(3)) > 0.96 and inside_wall
          and abs(float(p_b[1]) - ty_b) > c.tray_xy_tol
          and not bool(scene.tray_ok()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.40)

    # ========================= 8. negative: WRONG book in the tray ============================
    env.reset(seed=0)
    step(60)
    wrong = (tgt + 1) % c.n_books
    tp(scene.books[wrong], c.tray_x, tray_y,
       c.tray_floor_top + c.book_t[wrong] / 2 + 0.001, QX90)
    step(90)
    report("wrong-book")
    check("negative (wrong object): a NON-requested book in the tray, target still shelved — "
          "score ~0, no success",
          not bool(scene.others_ok()[0]) and not bool(scene.tray_ok()[0])
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]) and finite_all())

    # ========================= 9. negative: collateral (do-not-disturb) =======================
    env.reset(seed=0)
    step(60)
    other = (tgt + 2) % c.n_books
    tp(scene.books[tgt], c.tray_x, tray_y, c.tray_floor_top + t_t / 2 + 0.001, QX90)
    tp(scene.books[other], -0.03, -0.35, c.z0 + c.book_t[other] / 2 + 0.001, QX90)
    step(90)
    report("collateral")
    check("negative (collateral): target correct in tray BUT another book toppled out — "
          "no success, only the live tray credit",
          bool(scene.tray_ok()[0]) and not bool(scene.others_ok()[0])
          and not bool(scene.success()[0])
          and 0.25 < float(scene.score()[0]) < 0.40)

    # ========================= 10. exactness: goal state == score 1.0 =========================
    env.reset(seed=0)
    step(60)
    tp(scene.books[tgt], c.tray_x, tray_y, c.tray_floor_top + t_t / 2 + 0.01, QX90)
    good = False
    for _ in range(60):
        if bool(scene.success()[0]):
            good = True
            break
        step(10)
    report("goal-state")
    good = good and abs(float(scene.score()[0]) - 1.0) < 1e-6
    step(120)  # ...and it persists hands-off
    report("goal-state+1s")
    check("exactness: target flat in tray, row intact, settled -> success() and "
          "score == 1.0 exactly, stable",
          good and bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 1.0) < 1e-6)

    # ========================= 11. achievement latch (success is live) ========================
    tp(scene.books[tgt], c.book_x, float(scene.slot_y[0, tgt]), c.zs + h_t / 2 + 0.001, QID)
    step(90)
    report("revoked")
    check("achievement latch: book stood back in its slot revokes success; only the "
          "latched carry-out share remains",
          not bool(scene.success()[0]) and not bool(scene.tray_ok()[0])
          and 0.20 < float(scene.score()[0]) < 0.30)

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.librarian_extract")
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

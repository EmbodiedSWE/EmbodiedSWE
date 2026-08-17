"""Smoke battery for TiltShelfButtressScene — REJECTION tests for the rubric, NullRobot,
RECORDED.

This is NOT a solution (the solution is solve.py — bookend first, then four drop-into-
lean book placements plus a snug press; the Franka strategy is TASK.md's embodiment
argument). Wrong outcomes here are CONSTRUCTED as physical states (teleport + settle +
readback) and asserted rejected; the one success construct proves the rubric's ceiling
is exactly 1.0.

One linear run, 12 named checks:
  1. settle    — clean reset: finite state, four books physically upright AT their
                 sampled table slots and the bookend at its sampled pose (readback),
                 latches dark, score ~0, no success;
  2. readback  — the plank is physically at the sampled tilt: world quat == R_x(phi)
                 and position == (plank_x, plank_y, plank_z) (< 2 mm / < 0.5 deg),
                 tilt magnitude inside the cfg range;
  3. random    — 8 seeds: BOTH tilt signs occur, tilt magnitude and plank offset
                 vary, the table slot permutation varies; books at their slots on
                 every draw (readback);
  4. null      — 2 s of nothing: score < 0.05, no success;
  5. negative  — the SEED's plan (stand a book on the shelf, no anchor): a book
                 released standing on the BARE plank topples — settles far out of
                 the upright cone, score ~0, no success (the slope physically
                 defeats the seed strategy);
  6. negative  — full row WITHOUT the bookend: four books constructed in row poses
                 on the plank with the bookend left on the table — the row cascades,
                 nothing stands, chain 0, score ~0, no success (the anchor is
                 load-bearing);
  7. negative  — bookend at the WRONG (uphill) end with books downhill of it: books
                 collapse (nothing supports them), chain 0, no success, score <= the
                 bookend share (~0.25);
  8. negative  — books lying FLAT on the plank uphill of a correctly placed bookend:
                 stable but not standing — chain 0, no success, score ~0.25;
  9. negative  — lean near miss: one book leaning on the bookend's top edge at ~18
                 deg (> upright_max_deg=12): settled and stable but out of cone —
                 not counted, no success;
 10. exactness — the goal built physically (bookend drop + four drop-into-lean
                 placements) -> success() and score == 1.0 exactly, still true 1 s
                 later;
 11. latch     — removing the last book from the goal row revokes success (live
                 state) while the latched staged credit (~0.85) does NOT evaporate;
 12. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.put_books_on_bookshelf_i198.smoke --headless
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
    from simgen_tasks.put_books_on_bookshelf_i198 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

U_BUT = -0.13  # bookend uphill coordinate used by the constructs (same as solve.py)
CLEAR = 0.003
LEAN = 0.06
DROP = 0.006


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.tilt_shelf_buttress")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((0.78, 0.62, 1.10)) + o),
                                tuple(np.array((-0.12, 0.0, 0.55)) + o),
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
        bo = scene.books_ok()[0].tolist()
        print(f"[smoke] {tag:16s} | but_ok={bool(scene.buttress_ok()[0])} "
              f"books_ok={[int(x) for x in bo]} chain={float(scene.chain_count()[0]):.0f} "
              f"latch=({float(scene.but_latch[0]):.2f},{float(scene.k_latch[0]):.2f}) "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def finite_all() -> bool:
        bodies = list(scene.books) + [scene.buttress] + list(scene.kin_parts.values())
        return all(bool(torch.isfinite(b.data.root_state_w).all()) for b in bodies) \
            and bool(torch.isfinite(scene.score()).all())

    # --- plank-frame helpers (mirror solve.py; refreshed after every reset) -------------------
    fr = {}

    def read_frame() -> None:
        fr["s"] = float(scene.tilt_sign[0])
        fr["theta"] = float(scene.tilt[0])
        fr["phi"] = float(scene.phi[0])
        fr["ppos"] = scene.plank_pos[0].tolist()
        fr["cphi"] = math.cos(fr["phi"])
        fr["sphi"] = math.sin(fr["phi"])

    def tp_local(body, lx: float, u: float, lz: float, lean: float = 0.0,
                 q_local: tuple | None = None) -> None:
        """Teleport a body to plank-local (depth lx, uphill u, normal lz); orientation =
        plank frame + downhill lean, or an explicit local quat."""
        from isaaclab.utils.math import quat_mul

        s, cphi, sphi, ppos = fr["s"], fr["cphi"], fr["sphi"], fr["ppos"]
        ly = s * u
        pos = (ppos[0] + lx,
               ppos[1] + ly * cphi - lz * sphi,
               ppos[2] + ly * sphi + lz * cphi)
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor(pos, device=device) + scene.env_origins[0]
        if q_local is None:
            a = fr["phi"] + s * lean
            st[0, 3] = math.cos(a / 2)
            st[0, 4] = math.sin(a / 2)
        else:
            pq = torch.tensor([[math.cos(fr["phi"] / 2), math.sin(fr["phi"] / 2), 0.0, 0.0]],
                              device=device)
            st[0, 3:7] = quat_mul(pq, torch.tensor([q_local], device=device))[0]
        body.write_root_state_to_sim(st, ids0)

    def tp_table(body, x: float, y: float, z: float) -> None:
        st = torch.zeros(1, 13, device=device)
        st[0, 0:3] = torch.tensor((x, y, z), device=device) + scene.env_origins[0]
        st[0, 3] = 1.0
        body.write_root_state_to_sim(st, ids0)

    def u_of(body) -> float:
        p = body.data.root_pos_w[0] - scene.env_origins[0] - scene.plank_pos[0]
        return fr["s"] * (float(p[1]) * fr["cphi"] + float(p[2]) * fr["sphi"])

    def align_deg(body) -> float:
        from isaaclab.utils.math import quat_apply

        up = quat_apply(body.data.root_quat_w[:1],
                        torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
        nrm = scene.normal_w()[0]
        d = float((up * nrm).sum().clamp(-1.0, 1.0))
        return math.degrees(math.acos(d))

    def settle(max_steps: int = 420, thr: float = 0.03, need: int = 25) -> bool:
        streak = 0
        for _ in range(max_steps):
            step(1)
            _p, _q, lv, av = scene._body_tensors()
            if float(lv.max()) < thr and float(av.max()) < 0.4:
                streak += 1
                if streak >= need:
                    return True
            else:
                streak = 0
        return False

    def drop_book(k: int, face: float) -> float:
        """Drop book k with its base face CLEAR uphill of `face`, pre-leaned downhill —
        the solve.py placement. Returns the new uphill face position."""
        t, h = c.book_t[k], c.book_h[k]
        u_c = face + CLEAR + t / 2 + (h / 2) * math.sin(LEAN)
        lz = c.plank_size[2] / 2 + (h / 2) * math.cos(LEAN) + (t / 2) * math.sin(LEAN) + DROP
        tp_local(scene.books[k], 0.0, u_c, lz, lean=LEAN)
        settle()
        return u_of(scene.books[k]) + t / 2

    def build_goal() -> None:
        """Physically build the goal state: bookend drop, then four drop-into-lean books."""
        tp_local(scene.buttress, 0.0, U_BUT,
                 c.plank_size[2] / 2 + c.but_size[2] / 2 + DROP, 0.0)
        settle()
        face = u_of(scene.buttress) + c.but_size[1] / 2
        for k in range(c.n_books):
            face = drop_book(k, face)

    def start_pose_err() -> float:
        err = 0.0
        for b in range(c.n_books):
            p = scene.books[b].data.root_pos_w[0] - scene.env_origins[0]
            err = max(err, float((p[:2] - scene.book_start[0, b]).norm()))
        p = scene.buttress.data.root_pos_w[0] - scene.env_origins[0]
        return max(err, float((p[:2] - scene.but_start[0]).norm()))

    def books_upright_on_table() -> bool:
        from isaaclab.utils.math import quat_apply

        ok = True
        for b in range(c.n_books):
            up = quat_apply(scene.books[b].data.root_quat_w[:1],
                            torch.tensor([[0.0, 0.0, 1.0]], device=device))[0]
            ok &= float(up[2]) > 0.99
        return ok

    # ========================= 1. settle / clean-slate ========================================
    env.reset(seed=0)
    read_frame()
    step(90)
    report("reset")
    check("settle: clean reset (finite, books upright AT their table slots, bookend at its "
          "slot, score ~0)",
          finite_all() and books_upright_on_table() and start_pose_err() < 0.008
          and float(scene.but_latch[0]) == 0.0 and float(scene.k_latch[0]) == 0.0
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 2. plank pose readback =========================================
    pq_want = torch.tensor([math.cos(fr["phi"] / 2), math.sin(fr["phi"] / 2), 0.0, 0.0],
                           device=device)
    pq_got = scene.kin_parts["plank"].data.root_quat_w[0]
    q_dot = float((pq_want * pq_got).sum().abs())
    p_err = float((scene.kin_parts["plank"].data.root_pos_w[0] - scene.env_origins[0]
                   - scene.plank_pos[0]).norm())
    tilt_deg = math.degrees(fr["theta"])
    print(f"[smoke] plank readback: qdot={q_dot:.6f} perr={p_err * 1000:.2f}mm "
          f"tilt={tilt_deg:.2f}deg sign={fr['s']:+.0f}", flush=True)
    check("readback: plank physically at the sampled tilt/sign/offset (< 2 mm, < 0.5 deg), "
          "tilt inside the cfg range",
          q_dot > 0.99999 and p_err < 0.002
          and c.tilt_min_deg - 0.01 <= tilt_deg <= c.tilt_max_deg + 0.01)

    # ========================= 3. randomization across seeds ==================================
    signs, tilts, planks, perms = [], [], [], []
    ok_slots = True
    for sd in range(2, 10):
        env.reset(seed=sd)
        read_frame()
        step(30)
        signs.append(int(fr["s"]))
        tilts.append(round(math.degrees(fr["theta"]), 3))
        planks.append(round(fr["ppos"][1], 4))
        perms.append(tuple(int(i) for i in torch.argsort(scene.book_start[0, :, 1]).tolist()))
        ok_slots &= books_upright_on_table() and start_pose_err() < 0.010
    print(f"[smoke] draws: signs={signs} tilts={tilts}", flush=True)
    print(f"[smoke]        plank_y={planks} perms={perms}", flush=True)
    check("randomization is real (both tilt signs, tilt magnitude, plank offset, slot "
          "permutation all vary; readback holds)",
          len(set(signs)) == 2 and len(set(tilts)) >= 5 and max(tilts) - min(tilts) > 1.0
          and len(set(planks)) >= 5 and max(planks) - min(planks) > 0.03
          and len(set(perms)) >= 3 and ok_slots)

    # ========================= 4. null policy =================================================
    env.reset(seed=0)
    read_frame()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 5. negative: the SEED's plan (book on the bare shelf) ==========
    env.reset(seed=0)
    read_frame()
    step(30)
    t0, h0 = c.book_t[0], c.book_h[0]
    tp_local(scene.books[0], 0.0, 0.0, c.plank_size[2] / 2 + h0 / 2 + DROP, 0.0)
    settle(600)
    report("bare-plank")
    a0 = align_deg(scene.books[0])
    print(f"[smoke] bare-plank book align={a0:.1f}deg (upright cone is "
          f"{c.upright_max_deg:.0f}deg)", flush=True)
    check("negative (seed strategy): a book stood on the BARE plank topples — far out of "
          "the cone, score ~0, no success",
          a0 > 30.0 and not bool(scene.books_ok()[0, 0])
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]) and finite_all())

    # ========================= 6. negative: full row WITHOUT the bookend ======================
    env.reset(seed=0)
    read_frame()
    step(30)
    face = U_BUT + c.but_size[1] / 2  # where the bookend face WOULD be — but it stays on the table
    for k in range(c.n_books):
        t, h = c.book_t[k], c.book_h[k]
        u_c = face + CLEAR + t / 2
        tp_local(scene.books[k], 0.0, u_c, c.plank_size[2] / 2 + h / 2 + DROP, 0.0)
        face = u_c + t / 2
    settle(720)
    report("no-anchor")
    n_ok = int(scene.books_ok()[0].sum())
    check("negative (no anchor): the row without the bookend cascades — nothing stands, "
          "chain 0, score ~0, no success",
          n_ok == 0 and float(scene.chain_count()[0]) == 0.0
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]) and finite_all())

    # ========================= 7. negative: bookend at the WRONG (uphill) end =================
    env.reset(seed=0)
    read_frame()
    step(30)
    tp_local(scene.buttress, 0.0, +0.13, c.plank_size[2] / 2 + c.but_size[2] / 2 + DROP, 0.0)
    settle()
    face = -0.02  # books DOWNHILL of the bookend, leaning on nothing
    for k in range(c.n_books):
        t, h = c.book_t[k], c.book_h[k]
        tp_local(scene.books[k], 0.0, face - t / 2, c.plank_size[2] / 2 + h / 2 + DROP, 0.0)
        face = face - t - CLEAR
    settle(720)
    report("wrong-end")
    check("negative (wrong end): bookend uphill of the books — books collapse, chain 0, "
          "no success, only the bookend share",
          int(scene.books_ok()[0].sum()) == 0 and float(scene.chain_count()[0]) == 0.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.30)

    # ========================= 8. negative: books lying FLAT on the plank =====================
    env.reset(seed=0)
    read_frame()
    step(30)
    tp_local(scene.buttress, 0.0, U_BUT, c.plank_size[2] / 2 + c.but_size[2] / 2 + DROP, 0.0)
    settle()
    q_flat = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)  # R_y(90): height axis -> depth axis
    for k, u in enumerate((-0.05, 0.00, 0.05, 0.10)):
        tp_local(scene.books[k], 0.0, u, c.plank_size[2] / 2 + c.book_t[k] / 2 + DROP,
                 q_local=q_flat)
    settle(600)
    report("flat-books")
    check("negative (flat books): lying flat uphill of a correct bookend is stable but not "
          "standing — chain 0, no success, only the bookend share",
          bool(scene.buttress_ok()[0]) and int(scene.books_ok()[0].sum()) == 0
          and float(scene.chain_count()[0]) == 0.0 and not bool(scene.success()[0])
          and 0.20 <= float(scene.score()[0]) <= 0.30)

    # ========================= 9. negative: lean near miss (out of the cone) ==================
    env.reset(seed=0)
    read_frame()
    step(30)
    tp_local(scene.buttress, 0.0, U_BUT, c.plank_size[2] / 2 + c.but_size[2] / 2 + DROP, 0.0)
    settle()
    beta = math.radians(18.0)
    t0, h0 = c.book_t[0], c.book_h[0]
    but_face = u_of(scene.buttress) + c.but_size[1] / 2
    u0 = but_face + 0.032  # base corner well uphill: the book leans onto the bookend TOP edge
    u_c = u0 - (h0 / 2) * math.sin(beta) + (t0 / 2) * math.cos(beta)
    lz = c.plank_size[2] / 2 + (h0 / 2) * math.cos(beta) + (t0 / 2) * math.sin(beta) + 0.008
    tp_local(scene.books[0], 0.0, u_c, lz, lean=beta)
    settle(600)
    report("lean-miss")
    a_miss = align_deg(scene.books[0])
    print(f"[smoke] lean near miss: settled align={a_miss:.1f}deg", flush=True)
    check("negative (lean near miss): a book leaning on the bookend top edge beyond the "
          "12 deg cone is settled but NOT counted, no success",
          13.0 < a_miss < 45.0 and not bool(scene.books_ok()[0, 0])
          and float(scene.chain_count()[0]) == 0.0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.30)

    # ========================= 10. exactness: goal state == score 1.0 =========================
    env.reset(seed=0)
    read_frame()
    step(30)
    build_goal()
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
    check("exactness: bookend + four-book leaning row settled -> success() and score == 1.0 "
          "exactly, stable",
          good and bool(scene.success()[0]) and abs(float(scene.score()[0]) - 1.0) < 1e-6)

    # ========================= 11. achievement latch (success is live) ========================
    tp_table(scene.books[3], c.book_slot_x, 0.0, c.z0 + c.book_h[3] / 2 + 0.002)
    step(90)
    report("revoked")
    check("achievement latch: removing the last book revokes success (live) while the "
          "latched staged credit (~0.85) does not evaporate",
          not bool(scene.success()[0])
          and 0.83 <= float(scene.score()[0]) <= 0.87)

    # ========================= 12. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.tilt_shelf_buttress")
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

"""Smoke — REJECTION battery for ShapeSorterScene's rubric (sim_gen task draw_triangle_i1).

This is NOT a solution (solve.py, the teleport solution, proves the rubric ACCEPTS a
correct physically-settled outcome). Every probe here CONSTRUCTS a settled wrong outcome
by teleporting pieces (instrumentation only), lets physics settle it, and asserts the
rubric REJECTS it. The battery never constructs a success() state: the most any probe
reaches is ONE correctly posted piece (calibration), success always False.

Battery:
   1. settle/no-NaN     — reset layout settles finite, pieces at rest in the staging spots;
   2. reset-zero        — score ~0 on the fresh scene, nothing contained, no success;
   3. randomization     — READBACK across seeds: box pose (floor panel xy + yaw), the
                          color->compartment assignment (>= 2 distinct permutations, and
                          the PHYSICAL red rim strip position matches the assignment) and
                          piece poses all move; pieces spawn clear of the box every seed;
   4. null policy       — 240 idle steps -> score ~0, no success;
   5. state roundtrip   — set_state(get_state) restores piece poses (readback);
   6. on-lid near-miss  — the cube settled ON the lid right beside its own aperture ->
                          not contained (exact-top-below-lid gate), score ~0;
   7. attitude near-miss— the card lying FLAT centred over its own slot: 42 mm across a
                          34 mm slot bridges the opening and rests on the rims -> not
                          contained (right place, wrong attitude), score ~0;
   8. wrong compartment — the cube posted through the GREEN aperture: physically
                          CONTAINED in green's compartment (bin readback proves it) yet
                          counted False by the matching clause -> score ~0, no success;
   9. all-in-one        — all three pieces posted through the RED aperture (cube fits any
                          yaw, cylinder fits, the card fits end-on): only the cube counts
                          (0.25), never success;
  10. calibration       — the cylinder posted through its OWN aperture counts (0.25);
                          success still False with the other two pieces in staging;
  11. persistence       — 240 further idle steps: the posted cylinder stays counted
                          (containment credit does not evaporate under correct behavior);
  12. beside-box        — pieces set on the floor against the box wall at their assigned
                          compartments' positions ("right compartment, outside the box")
                          -> score ~0;
  13. seed-strategy     — the three pieces arranged as a triangle outline on the open
                          floor (the seed's drawn-triangle end state, made of pieces) ->
                          score ~0, no success;
  14. frames            — video frames recorded and saved to frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.draw_triangle_i1.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> empty
# frames. Disable the driver check.
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
    from simgen_tasks.draw_triangle_i1 import scene as scene_mod  # noqa: F401
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

DROP_CLEAR = 0.022


def main() -> None:  # noqa: PLR0915
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.shape_sorter")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(1, device=device)
    origin = scene.env_origins[0]

    # --- recording (viewport rgb annotator, the proven server mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = origin.detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.10, -0.90, 0.80)) + o),
                                tuple(np.array((0.22, 0.0, 0.04)) + o),
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

    def sc() -> float:
        return float(scene.score()[0])

    def piece_xy(i: int):
        p = scene.pieces[i].data.root_pos_w[0] - origin
        return float(p[0]), float(p[1])

    def piece_local(i: int) -> tuple[float, float]:
        """Piece centre in the box frame (readback)."""
        bxy, yaw = scene.box_frame()
        x, y = piece_xy(i)
        dx, dy = x - float(bxy[0, 0]), y - float(bxy[0, 1])
        by = float(yaw[0])
        return (math.cos(by) * dx + math.sin(by) * dy,
                -math.sin(by) * dx + math.cos(by) * dy)

    def put(i: int, x: float, y: float, z: float, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Teleport piece i to an env-local pose with zero velocity (instrumentation)."""
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1], st[0, 2] = x, y, z
        st[0, 3:7] = torch.tensor(quat, device=device)
        st[0, 0:3] += origin
        scene.pieces[i].write_root_state_to_sim(st, all_ids)

    def apt(color: int) -> tuple[float, float, float]:
        """World (x, y) of color c's aperture centre + box yaw (readback)."""
        bxy, yaw = scene.box_frame()
        by = float(yaw[0])
        ly = (int(scene.assign[0, color]) - 1) * c.bin_pitch
        return (float(bxy[0, 0]) - math.sin(by) * ly,
                float(bxy[0, 1]) + math.cos(by) * ly, by)

    drop_half = (c.cube_edge / 2, c.cyl_h / 2, c.card_size[2] / 2)

    def drop_through(i: int, color: int, end_on: bool = False) -> None:
        """Hover-release piece i above color c's aperture (probe uses the same honest
        mechanism as solve.py: release outside the box, physics threads the opening)."""
        x, y, by = apt(color)
        cy, sy = math.cos(by / 2), math.sin(by / 2)
        if end_on:  # card long-axis vertical: q = qz(by) * qy(90 deg)
            c45 = math.cos(math.pi / 4)
            q = (cy * c45, -sy * c45, cy * c45, sy * c45)
            half = c.card_size[0] / 2
        else:
            q = (cy, 0.0, 0.0, sy)
            half = drop_half[i]
        put(i, x, y, c.surface_z + c.lid_z1 + DROP_CLEAR + half, q)
        step(200)

    def report(tag: str) -> None:
        print(f"[smoke] {tag:16s} | {scene.status_report()}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    # =========================== 1-2. settle / no-NaN / reset-zero ==========================
    torch.manual_seed(11)
    env.reset()
    step(60)
    report("settle")
    finite = all(bool(torch.isfinite(p.data.root_state_w).all()) for p in scene.pieces)
    speeds = [float(p.data.root_lin_vel_w[0].norm()) for p in scene.pieces]
    check("settle: states finite, all pieces at rest after 60 steps",
          finite and max(speeds) < 0.02)
    check("reset-zero: score ~0, nothing contained, no success on the fresh scene",
          sc() <= 0.005 and int((scene.piece_bin()[0] >= 0).sum()) == 0
          and not bool(scene.success()[0]))

    # =========================== 3. randomization is real ===================================
    reads, assigns, rim_ok, clear_ok = [], [], [], []
    for s in (21, 22, 23, 24):
        torch.manual_seed(s)
        env.reset()
        step(4)
        bxy, yaw = scene.box_frame()
        row = [float(bxy[0, 0]), float(bxy[0, 1]), float(yaw[0])]
        for i in range(3):
            row += list(piece_xy(i))
        reads.append(row)
        assigns.append(tuple(scene.assign[0].tolist()))
        # PHYSICAL rim readback: the red xs strip's box-local y == red's compartment centre
        rp = (scene.panels["rim_red_0"].data.root_pos_w[0] - origin)
        dx, dy = float(rp[0]) - row[0], float(rp[1]) - row[1]
        by = float(yaw[0])
        ly = -math.sin(by) * dx + math.cos(by) * dy
        rim_ok.append(abs(ly - (assigns[-1][0] - 1) * c.bin_pitch) < 0.002)
        # every piece spawns clear of the box (box-local x beyond the outer wall)
        clear_ok.append(all(piece_local(i)[0] < -(c.bin_half + c.wall_t) - 0.02
                            for i in range(3)))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (box_xy+yaw, 3x piece_xy):\n{np.round(arr, 3)}"
          f"\n[smoke] assigns={assigns} rim_ok={rim_ok} clear={clear_ok}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: box pose (xy + yaw), the color->compartment permutation and "
          "every piece pose move across seeds (readback); rim strips physically track "
          "the assignment; pieces spawn clear of the box every seed",
          spread[0] > 0.01 and spread[1] > 0.01 and spread[2] > math.radians(2.0)
          and len(set(assigns)) >= 2 and all(rim_ok) and all(clear_ok)
          and all(max(spread[3 + 2 * i], spread[4 + 2 * i]) > 0.03 for i in range(3)))

    # =========================== 4. null policy ==============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: 240 idle steps -> score ~0, nothing contained, no success",
          sc() <= 0.005 and not bool(scene.success()[0]))

    # =========================== 5. state roundtrip ==========================================
    torch.manual_seed(41)
    env.reset()
    step(30)
    saved = scene.get_state(all_ids)
    p_before = [piece_xy(i) for i in range(3)]
    put(0, 0.0, 0.0, 0.10)
    put(2, 0.05, 0.05, 0.10)
    step(5)
    scene.set_state(saved, all_ids)
    step(1)
    p_after = [piece_xy(i) for i in range(3)]
    err = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(p_before, p_after))
    check("state roundtrip: set_state(get_state) restores piece poses (max drift < 5 mm)",
          err < 0.005)

    # =========================== 6. on-lid near-miss =========================================
    torch.manual_seed(51)
    env.reset()
    step(20)
    x, y, by = apt(0)
    # onto the solid rim strip beside the red aperture (offset along the box y axis)
    off = 0.046
    put(0, x - math.sin(by) * off, y + math.cos(by) * off,
        c.surface_z + c.lid_z1 + c.cube_edge / 2 + 0.002,
        (math.cos(by / 2), 0.0, 0.0, math.sin(by / 2)))
    step(100)
    report("on-lid")
    check("on-lid near-miss: the cube settled ON the lid beside its own aperture is not "
          "contained (top-below-lid gate) -> score ~0, no success",
          int(scene.piece_bin()[0, 0]) == -1 and sc() <= 0.005
          and not bool(scene.success()[0]))

    # =========================== 7. attitude near-miss (card bridges its slot) ===============
    torch.manual_seed(61)
    env.reset()
    step(20)
    x, y, by = apt(2)
    cy2, sy2 = math.cos(by / 2), math.sin(by / 2)
    c45 = math.cos(math.pi / 4)
    # lying FLAT (42 mm across the 34 mm slot) centred over the slot: bridges the opening
    put(2, x, y, c.surface_z + c.lid_z1 + c.card_size[1] / 2 + 0.002,
        (cy2 * c45, cy2 * c45, sy2 * c45, sy2 * c45))
    step(100)
    report("bridge")
    check("attitude near-miss: the card lying flat over its own slot bridges it (42 mm "
          "across 34 mm) and rests on the rims -> not contained, score ~0",
          int(scene.piece_bin()[0, 2]) == -1 and sc() <= 0.005
          and not bool(scene.success()[0]))

    # =========================== 8. wrong compartment ========================================
    torch.manual_seed(71)
    env.reset()
    step(20)
    drop_through(0, color=1)  # cube through the GREEN aperture (fits when yaw-aligned)
    report("wrong-bin")
    check("wrong compartment: the cube posted through the green aperture is physically "
          "contained in green's compartment (bin readback) yet counts False by the "
          "matching clause -> score ~0, no success",
          int(scene.piece_bin()[0, 0]) == int(scene.assign[0, 1]) and sc() <= 0.005
          and not bool(scene.success()[0]))

    # =========================== 9. all-in-one ===============================================
    torch.manual_seed(81)
    env.reset()
    step(20)
    drop_through(0, color=0)
    drop_through(1, color=0)
    drop_through(2, color=0, end_on=True)  # card fits the red hole end-on (18x42 cross)
    step(120)
    report("all-in-one")
    cnt = scene.counted()[0].tolist()
    check("all-in-one: all three pieces posted through the RED aperture -> only the cube "
          "counts (0.25), success False",
          bool(cnt[0]) and not cnt[1] and not cnt[2] and abs(sc() - 0.25) < 0.01
          and not bool(scene.success()[0]))

    # =========================== 10-11. calibration + persistence ============================
    torch.manual_seed(91)
    env.reset()
    step(20)
    drop_through(1, color=1)  # cylinder through its OWN aperture
    report("calibration")
    check("calibration: the cylinder posted through its own aperture COUNTS (score 0.25); "
          "success still False with the other pieces in staging",
          bool(scene.counted()[0, 1]) and abs(sc() - 0.25) < 0.01
          and not bool(scene.success()[0]))
    step(240)
    report("persistence")
    check("persistence: 240 further idle steps keep the posted cylinder counted "
          "(containment credit does not evaporate under correct behavior)",
          bool(scene.counted()[0, 1]) and abs(sc() - 0.25) < 0.01)

    # =========================== 12. beside-box ==============================================
    torch.manual_seed(101)
    env.reset()
    step(20)
    bxy, yaw = scene.box_frame()
    by = float(yaw[0])
    rest_z = (c.cube_edge / 2, c.cyl_h / 2, c.card_size[1] / 2)
    quats = ((math.cos(by / 2), 0.0, 0.0, math.sin(by / 2)),
             (1.0, 0.0, 0.0, 0.0),
             (math.cos(by / 2) * c45, math.cos(by / 2) * c45,
              math.sin(by / 2) * c45, math.sin(by / 2) * c45))
    for i in range(3):
        lx = -(c.bin_half + c.wall_t + 0.045)
        ly = (int(scene.assign[0, i]) - 1) * c.bin_pitch
        put(i, float(bxy[0, 0]) + math.cos(by) * lx - math.sin(by) * ly,
            float(bxy[0, 1]) + math.sin(by) * lx + math.cos(by) * ly,
            c.surface_z + rest_z[i] + 0.002, quats[i])
    step(100)
    report("beside-box")
    check("beside-box: pieces set on the floor against the box wall at their assigned "
          "compartments' positions -> outside the box, score ~0, no success",
          sc() <= 0.005 and int((scene.piece_bin()[0] >= 0).sum()) == 0
          and not bool(scene.success()[0]))

    # =========================== 13. seed-strategy ===========================================
    torch.manual_seed(111)
    env.reset()
    step(20)
    tri = ((0.10, 0.0), (-0.02, -0.09), (-0.02, 0.09))
    for i, (tx, ty) in enumerate(tri):
        put(i, tx, ty, c.surface_z + rest_z[i] + 0.002,
            (1.0, 0.0, 0.0, 0.0) if i != 2 else (c45, c45, 0.0, 0.0))
    step(100)
    report("seed-strategy")
    check("seed-strategy: the pieces arranged as a triangle outline on the open floor "
          "(the seed's drawn-triangle end state) -> score ~0, no success",
          sc() <= 0.005 and not bool(scene.success()[0]))

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.shape_sorter")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("frames: video frames recorded and saved to frames.npz", len(frames) > 20)

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

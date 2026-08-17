"""Smoke / rubric-rejection battery for ClocheServiceScene (sim_gen task
`libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i221`) — NullRobot, teleported
probe states (instrumentation, NOT a solution), RECORDED.

solve.py already proves the rubric ACCEPTS the correct outcome; this battery proves it
REJECTS the wrong ones:

  1. settle/no-NaN      — reset settles finite: cover SEATED on the plate, cake standing
                          on the floor beside it, score 0, no latches, no success;
  2. mass readback      — custom spawners author MassAPI mass; get_masses() must return
                          the authored plate/cover/cake masses;
  3. randomization      — READBACK across seeds: plate xy, cake xy AND which SIDE of the
                          plate the cake starts on, cover yaw all move;
  4. null-policy-fails  — 240 idle steps -> score ~0, cover still seated, no success;
  5-6. oracle x2 seeds  — uncover -> serve -> re-cover (all releases outside credit
                          bands, gravity finishes each placement) -> success, score 1.0,
                          PERSISTS over 240 further steps;
  7-8. monotonicity     — 0 (idle) < 0.15 (uncovered) < 0.40 (served) == 0.40 after the
                          cake is taken OFF the plate again (latched credit does not
                          evaporate) < 1.0 (covered); partial states never reach 1.0;
  9. negative A (seed)  — the seed's own end state: the cover flipped into BOWL pose
                          (opening up) resting on the plate with the cake dropped inside
                          it -> upright + cake-height clauses reject;
 10. negative B         — cake served on the plate but the cover left parked on the
                          floor -> no success (score stays at the latched 0.40);
 11. negative C         — cover seated on an EMPTY plate, cake elsewhere on the floor ->
                          no success;
 12. negative D (perch) — cake standing near the plate edge, cover dropped centered: its
                          rim lands ON the cake and cannot seat -> no success;
 13. negative E         — cake dropped on TOP of the seated cover (roof/knob) -> not on
                          the plate, not enclosed -> no success;
 14. negative F (order) — serving onto the COVERED plate: the cake dropped over the
                          seated cover is deflected — it can never end enclosed, the
                          cover stays seated, no success (the ordering is physically
                          forced, not rubric fiat).

Records video frames throughout and saves frames.npz in the CURRENT WORKING DIRECTORY.
Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i221.smoke --headless
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

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i221 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie if anything below stalls.
_wd = threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cloche_service")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.45, -0.75, 0.55)) + o),
                                tuple(np.array((0.30, 0.0, 0.06)) + o),
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

    def settle_until(pred, max_steps: int = 500, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def sc() -> float:
        return float(scene.score()[0])

    def report(tag: str) -> None:
        d_cp = float(scene._dxy(scene.cake, scene.plate)[0])
        d_cc = float(scene._dxy(scene.cake, scene.cloche)[0])
        rim_dz = float((scene.cloche_rim_z() - scene.plate_top_z())[0])
        print(f"[smoke] {tag:16s} on_plate={bool(scene.cake_on_plate()[0])} "
              f"seated={bool(scene.seated()[0])} enclosed={bool(scene.enclosed()[0])} "
              f"settled={bool(scene.settled()[0])} d(cake,plate)={d_cp:.3f} "
              f"d(cake,cover)={d_cc:.3f} rim_dz={rim_dz:+.4f} score={sc():.2f} "
              f"success={bool(scene.success()[0])} frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def tp(body, pos_env, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor(
            [float(v) for v in pos_env], device=device)
        st[:, 3:7] = torch.tensor([float(v) for v in quat], device=device)
        body.write_root_state_to_sim(st, all_ids)

    def plate_xy() -> tuple[float, float]:
        p = scene.plate.data.root_pos_w[0] - scene.env_origins[0]
        return float(p[0]), float(p[1])

    def plate_top() -> float:
        return float(scene.plate_top_z()[0] - scene.env_origins[0, 2])

    def park_cloche() -> None:
        px, py = plate_xy()
        tp(scene.cloche, [px, py + 0.32, c.cloche_wall_h / 2 + 0.006])
        step(25)  # settled() is vacuous at the teleport instant — force a landing window
        settle_until(lambda: bool(scene.settled()[0]))

    def drop_cake(dx: float = 0.0, dy: float = 0.0, dz: float = 0.050) -> None:
        px, py = plate_xy()
        tp(scene.cake, [px + dx, py + dy, plate_top() + c.cake_h / 2 + dz])
        step(25)
        settle_until(lambda: bool(scene.settled()[0]))

    def drop_cloche(dx: float = 0.0, dy: float = 0.0, rim_clear: float = 0.020,
                    quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        px, py = plate_xy()
        root_z = plate_top() + c.cake_h + rim_clear + c.cloche_wall_h / 2
        tp(scene.cloche, [px + dx, py + dy, root_z], quat)
        step(25)
        settle_until(lambda: bool(scene.settled()[0]))

    def oracle_run() -> None:
        park_cloche()
        drop_cake()
        settle_until(lambda: bool(scene.cake_on_plate()[0]) and bool(scene.settled()[0]))
        drop_cloche(dx=0.008, dy=-0.006)
        settle_until(lambda: bool(scene.success()[0]))

    def cake_z() -> float:
        return float((scene.cake.data.root_pos_w - scene.env_origins)[0, 2])

    # =========================== 1. settle / no-NaN ==============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.plate.data.root_state_w, scene.cloche.data.root_state_w,
                     scene.cake.data.root_state_w], dim=-1)
    check("settle: states finite, cover SEATED on the plate, cake standing on the floor "
          "beside it, everything settled",
          bool(torch.isfinite(st0).all()) and bool(scene.seated()[0])
          and cake_z() < 0.06 and float(scene._dxy(scene.cake, scene.plate)[0]) > c.plate_r
          and bool(scene.settled()[0]))
    check("settle: score 0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0])
          and not bool(scene._uncover_ever[0]) and not bool(scene._serve_ever[0]))

    # =========================== 2. mass readback ================================================
    reads = {name: float(b.root_physx_view.get_masses().reshape(-1)[0])
             for name, b in (("plate", scene.plate), ("cloche", scene.cloche),
                             ("cake", scene.cake))}
    print(f"[smoke] mass readback={ {k: round(v, 3) for k, v in reads.items()} }", flush=True)
    check("mass readback: custom-spawned plate/cover and the cake all run their "
          "authored masses",
          abs(reads["plate"] - c.plate_mass) < 1e-3
          and abs(reads["cloche"] - c.cloche_mass) < 1e-3
          and abs(reads["cake"] - c.cake_mass) < 1e-3)

    # =========================== 3. randomization is real ========================================
    from isaaclab.utils.math import quat_apply

    rows = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        px, py = plate_xy()
        ck = (scene.cake.data.root_pos_w - scene.env_origins)[0]
        ex = quat_apply(scene.cloche.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        rows.append((px, py, float(ck[0]), float(ck[1]),
                     1.0 if float(ck[1]) > 0 else -1.0, yaw))
    arr = np.array(rows)
    print(f"[smoke] randomization readback (plate_x, plate_y, cake_x, cake_y, side, "
          f"cloche_yaw):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: plate xy moves, cake xy moves, the cake's SIDE of the plate "
          "flips across seeds, cover yaw moves",
          (spread[0] > 0.01 or spread[1] > 0.01) and spread[2] > 0.015
          and len(set(arr[:, 4])) == 2 and spread[5] > 0.5)

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: 240 idle steps -> score ~0, cover still seated, no success",
          sc() <= 0.02 and bool(scene.seated()[0]) and not bool(scene.success()[0]))

    # =========================== 5-6. oracle on 2 seeds ==========================================
    for s in (0, 1):
        torch.manual_seed(s)
        env.reset()
        step(60)
        oracle_run()
        step(240)  # persistence: success must not flicker off
        report(f"oracle{s}-final")
        check(f"oracle seed {s}: uncover -> serve -> re-cover reaches success, score "
              f"1.0, persists 240 steps",
              bool(scene.success()[0]) and sc() == 1.0)

    # =========================== 7-8. rubric monotonicity ========================================
    torch.manual_seed(41)
    env.reset()
    step(60)
    s0 = sc()
    park_cloche()
    s_unc = sc()
    report("mono-uncovered")
    drop_cake()
    settle_until(lambda: bool(scene.cake_on_plate()[0]) and bool(scene.settled()[0]))
    s_srv = sc()
    report("mono-served")
    # take the cake OFF the plate again — latched credit must not evaporate
    px, py = plate_xy()
    tp(scene.cake, [px + 0.05, py - 0.30, c.cake_h / 2 + 0.004])
    step(25)
    settle_until(lambda: bool(scene.settled()[0]))
    s_off = sc()
    report("mono-cake-off")
    drop_cake()
    settle_until(lambda: bool(scene.cake_on_plate()[0]) and bool(scene.settled()[0]))
    drop_cloche(dx=0.008, dy=-0.006)
    settle_until(lambda: bool(scene.success()[0]))
    s_fin = sc()
    report("mono-final")
    print(f"[smoke] monotonicity ladder: idle={s0:.3f} uncovered={s_unc:.3f} "
          f"served={s_srv:.3f} cake-off={s_off:.3f} success={s_fin:.3f}", flush=True)
    check("monotonicity: 0 < uncovered (0.15) < served (0.40) == after un-serving "
          "(latched credit does not evaporate) < success (1.0)",
          s0 <= 0.005 and abs(s_unc - 0.15) < 0.01 and abs(s_srv - 0.40) < 0.01
          and abs(s_off - s_srv) < 1e-6 and s_fin == 1.0)
    check("monotonicity: partial states score < 1.0", max(s0, s_unc, s_srv, s_off) < 1.0)

    # =========================== 9. negative A: the seed's own end state =========================
    # "Put the black bowl on the plate": the cover flipped into BOWL pose (opening up,
    # standing on its knob) on the plate, with the cake inside the open bowl.
    torch.manual_seed(51)
    env.reset()
    step(60)
    px, py = plate_xy()
    bowl_root_z = plate_top() + c.knob_h + c.cloche_roof_t + c.cloche_wall_h / 2 + 0.003
    tp(scene.cloche, [px, py, bowl_root_z], quat=(0.0, 1.0, 0.0, 0.0))  # flipped 180 deg
    step(25)
    settle_until(lambda: bool(scene.settled()[0]))
    cavity_floor = plate_top() + c.knob_h + c.cloche_roof_t
    tp(scene.cake, [px, py, cavity_floor + c.cake_h / 2 + 0.030])
    step(25)
    settle_until(lambda: bool(scene.settled()[0]))
    step(120)
    report("seed-endstate")
    up_z = float(scene._up_of(scene.cloche)[0, 2])
    check("negative A (seed end state): cover in BOWL pose on the plate with the cake "
          "inside it -> upright clause and cake-height clause both reject, no success",
          up_z < 0.0 and not bool(scene.seated()[0])
          and not bool(scene.cake_on_plate()[0]) and not bool(scene.success()[0]))

    # =========================== 10. negative B: served but never re-covered =====================
    torch.manual_seed(61)
    env.reset()
    step(60)
    park_cloche()
    drop_cake()
    settle_until(lambda: bool(scene.cake_on_plate()[0]) and bool(scene.settled()[0]))
    step(120)
    report("no-recover")
    check("negative B (cover left off): cake standing on the plate, cover parked on the "
          "floor -> no success, score stays at the latched 0.40",
          bool(scene.cake_on_plate()[0]) and not bool(scene.success()[0])
          and abs(sc() - 0.40) < 0.01)

    # =========================== 11. negative C: covered but never served ========================
    torch.manual_seed(71)
    env.reset()
    step(60)
    px, py = plate_xy()
    tp(scene.cake, [px - 0.05, py - 0.34, c.cake_h / 2 + 0.004])  # cake elsewhere on the floor
    step(25)
    settle_until(lambda: bool(scene.settled()[0]))
    step(120)
    report("empty-covered")
    check("negative C (nothing served): cover seated on the EMPTY plate, cake on the "
          "floor -> no success",
          bool(scene.seated()[0]) and not bool(scene.cake_on_plate()[0])
          and not bool(scene.success()[0]))

    # =========================== 12. negative D: cover perched on the cake =======================
    torch.manual_seed(81)
    env.reset()
    step(60)
    park_cloche()
    drop_cake(dx=0.068)  # standing on the plate, under where the falling rim will land
    on_plate_before = bool(scene.cake_on_plate()[0])
    drop_cloche()  # centered: its rim comes down ON the cake
    step(120)
    report("perched")
    check("negative D (perch): cover dropped onto a cake standing at the rim line "
          "cannot seat (tilted / high rim) -> no success",
          on_plate_before and not bool(scene.success()[0]))

    # =========================== 13. negative E: cake on top of the cover ========================
    torch.manual_seed(91)
    env.reset()
    step(60)
    px, py = plate_xy()
    roof_top = plate_top() + c.cloche_wall_h + c.cloche_roof_t
    tp(scene.cake, [px + 0.03, py, roof_top + c.cake_h / 2 + 0.015])
    step(25)
    settle_until(lambda: bool(scene.settled()[0]))
    step(120)
    report("cake-on-roof")
    check("negative E (cake atop the cover): cake dropped onto the seated cover's top "
          "-> never on the plate top, no success",
          not bool(scene.cake_on_plate()[0]) and not bool(scene.success()[0]))

    # =========================== 14. negative F: serving onto the covered plate ==================
    # The ordering is physically forced: with the cover seated, a cake dropped over the
    # plate is deflected by the cover (and the plate annulus beside the cover is too
    # narrow to hold it) — it can never end enclosed, and the cover stays seated.
    torch.manual_seed(101)
    env.reset()
    step(60)
    px, py = plate_xy()
    rel = (px + 0.02, py + 0.015)
    tp(scene.cake, [rel[0], rel[1],
                    plate_top() + c.cloche_wall_h + c.cloche_roof_t + c.cake_h / 2 + 0.06])
    step(25)
    settle_until(lambda: bool(scene.settled()[0]))
    step(120)
    report("serve-covered")
    ck = (scene.cake.data.root_pos_w - scene.env_origins)[0]
    moved_off_axis = float(scene._dxy(scene.cake, scene.cloche)[0]) > c.enc_tol
    fell = float(ck[2]) < plate_top() + c.cloche_wall_h  # deflected below the roof line
    check("negative F (forced order): a cake dropped onto the COVERED plate is "
          "deflected — not enclosed, cover still seated, no success",
          moved_off_axis and fell and bool(scene.seated()[0])
          and not bool(scene.enclosed()[0]) and not bool(scene.success()[0]))

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cloche_service")
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
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL exception", flush=True)
        os._exit(2)

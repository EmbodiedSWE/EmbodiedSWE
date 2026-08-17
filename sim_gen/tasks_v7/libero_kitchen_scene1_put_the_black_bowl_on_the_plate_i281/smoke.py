"""Smoke / rubric-rejection battery for BerryPourScene (sim_gen task
`libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i281`) — NullRobot, teleported
probe states (instrumentation, NOT a solution), RECORDED.

solve.py already proves the rubric ACCEPTS the correct outcome (poured berries on the
plate, bowl set back down clear); this battery proves it REJECTS the wrong ones:

  1. settle/no-NaN      — reset settles finite: every present berry INSIDE the bowl,
                          bowl upright on the floor clear of the plate, score 0, no
                          latches, no success;
  2. mass readback      — custom spawners author MassAPI mass; get_masses() must return
                          the authored plate/bowl/berry masses;
  3. randomization      — READBACK across seeds: plate xy, bowl xy, bowl yaw, and the
                          sampled BERRY COUNT all move;
  4. null-policy-fails  — 240 idle steps -> score ~0, berries still in the bowl,
                          no success;
  5-6. deliver latch    — two berries settled on the plate score exactly the fractional
                          credit (no lift credit), below 1.0; pulling one back OFF the
                          plate leaves the latched score unchanged (credit does not
                          evaporate under regression);
  7. negative A (seed)  — the SEED's own end state: the bowl, still holding its
                          berries, resting ON the plate -> the in-bowl exclusion counts
                          zero berries, the bowl-clear clause fails, no success;
  8. negative B         — all berries genuinely on the plate but the emptied bowl LEFT
                          ON the plate -> bowl-clear rejects, no success;
  9. negative C         — berries on the plate, bowl off the plate but lying on its
                          SIDE -> upright clause rejects, no success;
 10. negative D (miss)  — one berry settled on the floor just outside the plate, the
                          rest on it, bowl parked correctly -> not all delivered,
                          no success;
 11. negative E (dump)  — berries dumped on the floor beside the bowl, none on the
                          plate -> zero counted, score ~0, no success;
 12. rim-hug honesty    — a berry that rolls against the plate rim's INNER face still
                          counts (the containment-vs-corner rule), so correct pours
                          near the rim are never rejected.

Records video frames throughout and saves frames.npz in the CURRENT WORKING DIRECTORY.
Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i281.smoke --headless
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
    from simgen_tasks.libero_kitchen_scene1_put_the_black_bowl_on_the_plate_i281 import (  # noqa: F401,E501
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
    env = ENVS.get("simgen.berry_pour")().build(num_envs=args.num_envs, device=device)
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
                                tuple(np.array((0.25, 0.0, 0.05)) + o),
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

    def n_pres() -> int:
        return int(scene.n_present()[0])

    def n_counted() -> int:
        return int(scene.counted()[0].sum())

    def n_in_bowl() -> int:
        return int((scene.in_bowl() & scene.present)[0].sum())

    def report(tag: str) -> None:
        print(f"[smoke] {tag:16s} present={n_pres()} on_plate={n_counted()} "
              f"in_bowl={n_in_bowl()} bowl_clear={bool(scene.bowl_clear()[0])} "
              f"upright={bool(scene.bowl_upright()[0])} settled={bool(scene.settled()[0])} "
              f"score={sc():.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

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

    def bowl_xy() -> tuple[float, float]:
        p = scene.bowl.data.root_pos_w[0] - scene.env_origins[0]
        return float(p[0]), float(p[1])

    def present_ids() -> list[int]:
        return [i for i in range(c.berry_max) if bool(scene.present[0, i])]

    def grid_slot(j: int) -> tuple[float, float]:
        """Berry drop slots over the plate: center, then a ring well inside the rim."""
        if j == 0:
            return (0.0, 0.0)
        ang = 2 * math.pi * (j - 1) / 5
        return (0.05 * math.cos(ang), 0.05 * math.sin(ang))

    def drop_berries_on_plate(ids: list[int]) -> None:
        px, py = plate_xy()
        for j, i in enumerate(ids):
            dx, dy = grid_slot(j)
            tp(scene.berries[i], [px + dx, py + dy, plate_top() + c.berry_r + 0.012])
            step(10)
        step(30)
        settle_until(lambda: bool(scene.settled()[0]))

    def park_bowl_clear(dy: float = -0.30) -> None:
        """Bowl upright on the floor, well clear of the plate (its own reset home is
        clear too; this spot survives constructs that need the home area)."""
        px, py = plate_xy()
        tp(scene.bowl, [px - 0.32, py + dy, c.bowl_rest_z + 0.003])
        step(25)
        settle_until(lambda: bool(scene.settled()[0]))

    def seat_berries_in_bowl(ids: list[int]) -> None:
        """Re-seat berries inside the bowl at its CURRENT location (constructs the
        'bowl still loaded' states)."""
        bx, by = bowl_xy()
        bz = float((scene.bowl.data.root_pos_w[0] - scene.env_origins[0])[2])
        floor_z = bz - c.bowl_wall_h / 2
        for j, i in enumerate(ids):
            layer, slot = divmod(j, 3)
            ang = 2 * math.pi * slot / 3 + layer * (math.pi / 3)
            tp(scene.berries[i],
               [bx + 0.024 * math.cos(ang), by + 0.024 * math.sin(ang),
                floor_z + c.berry_r + 0.004 + layer * (2 * c.berry_r + 0.004)])
            step(4)
        step(30)
        settle_until(lambda: bool(scene.settled()[0]))

    # =========================== 1. settle / no-NaN ==============================================
    torch.manual_seed(11)
    env.reset()
    report("reset")
    step(90)
    report("settled")
    st0 = torch.cat([scene.plate.data.root_state_w, scene.bowl.data.root_state_w]
                    + [b.data.root_state_w for b in scene.berries], dim=-1)
    check("settle: states finite, every present berry inside the bowl, bowl upright on "
          "the floor clear of the plate, everything settled",
          bool(torch.isfinite(st0).all()) and n_in_bowl() == n_pres()
          and bool(scene.bowl_clear()[0]) and bool(scene.settled()[0]))
    check("settle: score 0 at reset, no success, no latches",
          sc() <= 0.005 and not bool(scene.success()[0])
          and not bool(scene._lift_ever[0]) and float(scene._deliver_frac_max[0]) == 0.0)

    # =========================== 2. mass readback ================================================
    reads = {name: float(b.root_physx_view.get_masses().reshape(-1)[0])
             for name, b in (("plate", scene.plate), ("bowl", scene.bowl),
                             ("berry", scene.berries[0]))}
    print(f"[smoke] mass readback={ {k: round(v, 4) for k, v in reads.items()} }", flush=True)
    check("mass readback: custom-spawned plate/bowl and the berries all run their "
          "authored masses",
          abs(reads["plate"] - c.plate_mass) < 1e-3
          and abs(reads["bowl"] - c.bowl_mass) < 1e-3
          and abs(reads["berry"] - c.berry_mass) < 1e-3)

    # =========================== 3. randomization is real ========================================
    from isaaclab.utils.math import quat_apply

    rows = []
    for s in (21, 22, 23, 24, 25, 26):
        torch.manual_seed(s)
        env.reset()
        step(4)
        px, py = plate_xy()
        bx, by = bowl_xy()
        ex = quat_apply(scene.bowl.data.root_quat_w[0:1],
                        torch.tensor([[1.0, 0.0, 0.0]], device=device))[0]
        yaw = math.atan2(float(ex[1]), float(ex[0]))
        # count by READBACK: berries physically near the bowl (not in the depot)
        pos = scene._berry_pos()[0] - scene.env_origins[0]
        near = int(((pos[:, :2] - torch.tensor([bx, by], device=device)).norm(dim=-1)
                    < 0.30).sum())
        rows.append((px, py, bx, by, yaw, near))
    arr = np.array(rows)
    print(f"[smoke] randomization readback (plate_x, plate_y, bowl_x, bowl_y, bowl_yaw, "
          f"count):\n{np.round(arr, 3)}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: plate xy moves, bowl xy moves, bowl yaw moves, the sampled "
          "berry COUNT varies across seeds",
          spread[0] > 0.01 and spread[2] > 0.015 and spread[4] > 0.5
          and len(set(arr[:, 5])) >= 2
          and all(c.min_present <= v <= c.berry_max for v in arr[:, 5]))

    # =========================== 4. null policy fails ============================================
    torch.manual_seed(31)
    env.reset()
    step(240)
    report("null-policy")
    check("null policy: 240 idle steps -> score ~0, berries still in the bowl, no success",
          sc() <= 0.02 and n_in_bowl() == n_pres() and not bool(scene.success()[0]))

    # =========================== 5-6. deliver latch + no evaporation =============================
    torch.manual_seed(41)
    env.reset()
    step(60)
    ids = present_ids()
    k = len(ids)
    drop_berries_on_plate(ids[:2])
    step(60)
    report("two-delivered")
    want = 0.55 * 2 / k
    s_two = sc()
    check("deliver latch: two berries settled on the plate score exactly the "
          "fractional credit (no lift credit), below 1.0",
          abs(s_two - want) < 0.02 and not bool(scene._lift_ever[0]) and s_two < 0.99
          and not bool(scene.success()[0]))
    # pull one delivered berry back off the plate — latched credit must not evaporate
    px, py = plate_xy()
    tp(scene.berries[ids[0]], [px - 0.05, py - 0.35, c.berry_r + 0.003])
    step(25)
    settle_until(lambda: bool(scene.settled()[0]))
    step(30)
    report("one-regressed")
    check("deliver latch: removing a delivered berry does not reduce the latched score",
          abs(sc() - s_two) < 1e-6 and n_counted() == 1)

    # =========================== 7. negative A: the seed's own end state =========================
    # "Put the black bowl on the plate": the loaded bowl carried onto the plate.
    torch.manual_seed(51)
    env.reset()
    step(60)
    ids = present_ids()
    px, py = plate_xy()
    tp(scene.bowl, [px, py, plate_top() + c.bowl_rest_z + 0.002])
    step(20)
    seat_berries_in_bowl(ids)
    step(120)
    report("seed-endstate")
    check("negative A (seed end state): the loaded bowl resting ON the plate -> its "
          "berries count zero (in-bowl exclusion), bowl-clear fails, no success",
          n_counted() == 0 and n_in_bowl() == len(ids)
          and not bool(scene.bowl_clear()[0]) and not bool(scene.success()[0])
          and sc() <= 0.11)

    # =========================== 8. negative B: emptied bowl left on the plate ===================
    torch.manual_seed(61)
    env.reset()
    step(60)
    ids = present_ids()
    px, py = plate_xy()
    # berries genuinely on the plate on one side, the bowl resting on the other side
    for j, i in enumerate(ids):
        ang = math.pi * 0.75 + (math.pi * 0.5) * (j / max(len(ids) - 1, 1))
        tp(scene.berries[i], [px + 0.075 * math.cos(ang), py + 0.075 * math.sin(ang),
                              plate_top() + c.berry_r + 0.012])
        step(8)
    tp(scene.bowl, [px + 0.038, py, plate_top() + c.bowl_rest_z + 0.002])
    step(30)
    settle_until(lambda: bool(scene.settled()[0]))
    step(120)
    report("bowl-on-plate")
    check("negative B (bowl left on the plate): all berries on the plate but the bowl "
          "resting on it too -> bowl-clear rejects, no success",
          n_counted() >= 1 and not bool(scene.bowl_clear()[0])
          and not bool(scene.success()[0]))

    # =========================== 9. negative C: bowl tipped on its side ==========================
    torch.manual_seed(71)
    env.reset()
    step(60)
    ids = present_ids()
    drop_berries_on_plate(ids)
    px, py = plate_xy()
    tp(scene.bowl, [px - 0.34, py - 0.20, c.bowl_outer_r + 0.004],
       quat=(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0))  # lying on its side
    step(30)
    settle_until(lambda: bool(scene.settled()[0]))
    step(120)
    report("bowl-on-side")
    delivered_all = bool(scene.all_delivered()[0])
    check("negative C (bowl tipped over): every berry on the plate but the bowl lying "
          "on its side off the plate -> upright clause rejects, no success",
          delivered_all and not bool(scene.bowl_upright()[0])
          and not bool(scene.bowl_clear()[0]) and not bool(scene.success()[0]))

    # =========================== 10. negative D: one berry missed the plate ======================
    torch.manual_seed(81)
    env.reset()
    step(60)
    ids = present_ids()
    drop_berries_on_plate(ids[1:])
    px, py = plate_xy()
    # the missed berry: settled on the floor just outside the plate edge
    tp(scene.berries[ids[0]], [px + c.plate_r + 0.03, py, c.berry_r + 0.003])
    park_bowl_clear()
    step(120)
    report("one-missed")
    check("negative D (near miss): one berry on the floor beside the plate, the rest "
          "on it, bowl parked correctly -> not all delivered, no success",
          n_counted() == len(ids) - 1 and not bool(scene.all_delivered()[0])
          and bool(scene.bowl_clear()[0]) and not bool(scene.success()[0]))

    # =========================== 11. negative E: dumped on the floor =============================
    torch.manual_seed(91)
    env.reset()
    step(60)
    ids = present_ids()
    bx, by = bowl_xy()
    for j, i in enumerate(ids):
        ang = 2 * math.pi * j / len(ids)
        tp(scene.berries[i], [bx + 0.12 * math.cos(ang), by + 0.12 * math.sin(ang),
                              c.berry_r + 0.005])
        step(6)
    step(30)
    settle_until(lambda: bool(scene.settled()[0]))
    step(120)
    report("floor-dump")
    check("negative E (floor dump): berries tipped out on the floor beside the bowl -> "
          "zero counted, score ~0, no success",
          n_counted() == 0 and sc() <= 0.02 and not bool(scene.success()[0]))

    # =========================== 12. rim-hug honesty =============================================
    torch.manual_seed(101)
    env.reset()
    step(60)
    ids = present_ids()
    px, py = plate_xy()
    # released just inside the rim: it rolls out and rests AGAINST the rim's inner face
    tp(scene.berries[ids[0]], [px + c.plate_rim_inner_r - c.berry_r - 0.004, py,
                               plate_top() + c.berry_r + 0.012])
    step(30)
    settle_until(lambda: bool(scene.settled()[0]))
    step(60)
    dxy = float((scene._berry_pos()[0, ids[0], :2]
                 - scene.plate.data.root_pos_w[0, :2]).norm())
    report("rim-hug")
    check("rim-hug honesty: a berry resting against the plate rim's inner face still "
          "counts as on the plate",
          dxy > 0.080 and bool(scene.counted()[0, ids[0]]))

    # =========================== save + verdict ==================================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.berry_pour")
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

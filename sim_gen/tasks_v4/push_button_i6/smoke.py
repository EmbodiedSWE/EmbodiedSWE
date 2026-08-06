"""smoke — REJECTION battery for the WeighbridgeScene rubric (NullRobot, RECORDED).

This module is NOT a solution (solve.py — the real Franka run — already proves the
rubric ACCEPTS the correct outcome). Every check here CONSTRUCTS a wrong outcome as a
settled/pinned state (teleports are instrumentation) and asserts the rubric REJECTS it:

  1. settle/no-NaN       — reset layout settles finite, plate at home on its axis, score 0;
  2. randomization       — READBACK: block slots/jitter/yaw differ across seeded resets;
  3. null policy         — 240 idle steps -> score ~0, no success;
  4. seed strategy       — the seed's plan (a sustained bare press: the plate kinematically
                           held at depth 150 substeps, nothing resting on it) NEVER succeeds;
  5. spring-back         — releasing that press returns the plate home (nothing latches);
  6. press-through-foam  — an "infinitely patient" press THROUGH a 60 g foam block pinned at
                           depth: on-plate mass 0.06 kg < 0.29 kg trigger -> never success;
  7. resting foam        — a foam resting centred on the plate sags it only ~7 mm: partial
                           credit only, never success;
  8. stacked foams       — both foams stacked on the plate still stay far under the press
                           threshold: no success;
  9. wrong place A       — the load block settled on the GROUND beside the pedestal: score ~0;
 10. wrong place B       — the load block on the PEDESTAL RIM beside the plate: no press,
                           no success;
 11. off-centre          — the load block released just OUTSIDE the on-plate radius (straddling
                           the plate edge): never success;
 12. staged partials     — lifted-only < foam-on-plate partial < 1.0, monotone.

Records video frames -> frames.npz in the CWD.
Run (forge): python -u -m simgen_tasks.push_button_i6.smoke --headless
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
# RTX recipe: kit mis-decodes the L20 driver version and silently rejects RTX -> the
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
from simgen_tasks.push_button_i6 import scene as scene_mod  # noqa: F401, E402


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weighbridge")().build(num_envs=1, device=device)
    scene = env.scene
    c = scene.cfg
    no_action = torch.empty(0, device=device)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    edges = {n: e for n, e, _m, _rgb in c.blocks}

    # --- recording (viewport rgb annotator, the proven forge mechanism) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.55, -1.55, 0.95)) + o),
                                tuple(np.array((0.02, 0.0, 0.08)) + o),
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

    def report(tag: str) -> None:
        print(f"[smoke] {tag:18s} depth={float(scene.plate_depth()[0])*1000:5.1f}mm "
              f"m_on={float(scene.loaded_mass()[0]):.3f}kg "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def block_local(name: str) -> torch.Tensor:
        return (scene.blocks[name].data.root_pos_w - scene.env_origins)[0]

    def put_block(name: str, dxy: tuple, z_gap: float = 0.004, settle: int = 240,
                  on: str = "plate") -> None:
        """Teleport a block to rest z_gap above the plate top (on='plate'), the pedestal
        top (on='ped'), the ground (on='ground'), or on another block (on=<block>)."""
        edge = edges[name]
        st = torch.zeros(1, 13, device=device)
        if on == "plate":
            pl = scene.plate.data.root_pos_w[0]
            st[0, 0], st[0, 1] = pl[0] + dxy[0], pl[1] + dxy[1]
            st[0, 2] = pl[2] + c.plate_size[2] / 2 + edge / 2 + z_gap
        elif on == "ped":
            pd = scene.pedestal.data.root_pos_w[0]
            st[0, 0], st[0, 1] = pd[0] + dxy[0], pd[1] + dxy[1]
            st[0, 2] = pd[2] + c.ped_size[2] / 2 + edge / 2 + z_gap
        elif on == "ground":
            st[0, 0], st[0, 1] = dxy[0] + scene.env_origins[0, 0], dxy[1] + scene.env_origins[0, 1]
            st[0, 2] = scene.env_origins[0, 2] + c.surface_z + edge / 2 + z_gap
        else:
            bp = scene.blocks[on].data.root_pos_w[0]
            st[0, 0], st[0, 1] = bp[0] + dxy[0], bp[1] + dxy[1]
            st[0, 2] = bp[2] + edges[on] / 2 + edge / 2 + z_gap
        st[0, 3] = 1.0
        scene.blocks[name].write_root_state_to_sim(st, ids)
        step(settle)

    def pin_plate(depth: float, n_sub: int, foam: str | None = None) -> None:
        """Kinematically hold the plate at `depth` for n_sub substeps — the SEED-strategy
        probe (a patient press). With `foam`, the foam block is pinned resting on the
        pressed plate too (a press THROUGH a too-light block)."""
        for _ in range(n_sub):
            st = torch.zeros(1, 13, device=device)
            st[0, 0:2] = scene.pedestal.data.root_pos_w[0, 0:2]
            st[0, 2] = scene._home_z[0] - depth
            st[0, 3] = 1.0
            scene.plate.write_root_state_to_sim(st, ids)
            if foam is not None:
                fs = torch.zeros(1, 13, device=device)
                fs[0, 0:2] = st[0, 0:2]
                fs[0, 2] = st[0, 2] + c.plate_size[2] / 2 + edges[foam] / 2
                fs[0, 3] = 1.0
                scene.blocks[foam].write_root_state_to_sim(fs, ids)
            step(1)

    # =========================== 1. settle / no-NaN =========================================
    env.reset(seed=11)
    step(90)
    report("reset")
    pp = (scene.plate.data.root_pos_w - scene.env_origins)[0]
    pd = (scene.pedestal.data.root_pos_w - scene.env_origins)[0]
    check("settle: all states finite, plate at home on the pedestal axis",
          bool(torch.isfinite(scene.plate.data.root_state_w).all())
          and float(scene.plate_depth()[0]) < 0.003
          and float((pp[:2] - pd[:2]).norm()) < 0.003)
    check("settle: score ~0 at reset", float(scene.score()[0]) <= 0.005)

    # =========================== 2. randomization is real ===================================
    reads = []
    for s in (21, 22, 23, 24):
        env.reset(seed=s)
        step(5)
        lp = block_local("load")
        fp = block_local("foam_0")
        q = scene.blocks["load"].data.root_quat_w[0]
        yaw = math.atan2(2 * float(q[0] * q[3] + q[1] * q[2]),
                         1 - 2 * float(q[2] * q[2] + q[3] * q[3]))
        reads.append((float(lp[0]), float(lp[1]), float(fp[0]), float(fp[1]), yaw))
    arr = np.array(reads)
    print(f"[smoke] randomization readback (load_x, load_y, foam0_x, foam0_y, load_yaw):\n"
          f"{arr}", flush=True)
    spread = arr.max(axis=0) - arr.min(axis=0)
    check("randomization: block positions and yaw differ across seeded resets (readback)",
          (spread[0] + spread[1]) > 0.05 and (spread[2] + spread[3]) > 0.05
          and spread[4] > 0.2)

    # =========================== 3. null policy fails =======================================
    env.reset(seed=31)
    step(240)
    report("null-policy")
    check("null policy: score ~0 and no success after 240 idle steps",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # =========================== 4./5. seed strategy: sustained bare press ==================
    env.reset(seed=41)
    step(60)
    pin_plate(0.034, 150)
    report("bare-press")
    check("seed strategy: a sustained bare press (150 substeps at full depth, nothing "
          "resting) NEVER succeeds, score ~0",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02)
    step(120)
    report("press-released")
    check("spring-back: plate returns home once the press is released, still no credit",
          float(scene.plate_depth()[0]) < 0.005 and float(scene.score()[0]) <= 0.02
          and not bool(scene.success()[0]))

    # =========================== 6. press THROUGH a foam block ==============================
    env.reset(seed=51)
    step(60)
    pin_plate(0.034, 150, foam="foam_0")
    report("press-thru-foam")
    check("insufficient mass: a patient press THROUGH a 60 g foam (plate held at depth "
          "150 substeps, foam resting on it) never succeeds",
          not bool(scene.success()[0]) and float(scene.score()[0]) < 1.0)

    # =========================== 7. resting foam near-miss ==================================
    env.reset(seed=61)
    step(60)
    put_block("foam_0", (0.0, 0.0))
    report("foam-resting")
    sag = float(scene.plate_depth()[0])
    check("near-miss: one resting foam sags the plate only partially (3..15 mm), "
          "partial credit, no success",
          0.003 < sag < 0.015 and not bool(scene.success()[0])
          and 0.4 < float(scene.score()[0]) < 1.0)

    # =========================== 8. both foams stacked ======================================
    env.reset(seed=71)
    step(60)
    put_block("foam_0", (0.0, 0.0))
    put_block("foam_1", (0.0, 0.0), on="foam_0", settle=300)
    report("foams-stacked")
    check("near-miss: both foams stacked on the plate stay under the press threshold, "
          "no success",
          float(scene.plate_depth()[0]) < c.press_depth and not bool(scene.success()[0])
          and float(scene.score()[0]) < 1.0)

    # =========================== 9. load on the ground beside ===============================
    env.reset(seed=81)
    step(60)
    pdp = (scene.pedestal.data.root_pos_w - scene.env_origins)[0]
    put_block("load", (float(pdp[0]), float(pdp[1]) - 0.28), on="ground", settle=240)
    report("load-beside")
    check("wrong place: the load block settled on the ground beside the pedestal "
          "scores ~0, no success",
          not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.02)

    # =========================== 10. load on the pedestal rim ===============================
    env.reset(seed=91)
    step(60)
    put_block("load", (c.plate_size[0] / 2 + edges["load"] / 2 + 0.004, 0.0),
              on="ped", settle=300)
    report("load-on-rim")
    check("wrong place: the load block on the pedestal rim beside the plate presses "
          "nothing, no success",
          float(scene.plate_depth()[0]) < c.press_depth and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.25)

    # =========================== 11. off-centre release =====================================
    env.reset(seed=101)
    step(60)
    put_block("load", (c.on_plate_r + 0.012, 0.0), settle=360)
    lp = block_local("load")
    report("load-off-centre")
    off = float((scene.blocks["load"].data.root_pos_w[0, :2]
                 - scene.plate.data.root_pos_w[0, :2]).norm())
    check("near-miss: load released just outside the on-plate radius (straddles the "
          "edge) never succeeds",
          not bool(scene.success()[0]))
    print(f"[smoke]   off-centre outcome: off_axis={off*1000:.0f}mm "
          f"z={float(lp[2]):.3f} depth={float(scene.plate_depth()[0])*1000:.1f}mm", flush=True)

    # =========================== 12. staged partials, monotone ==============================
    env.reset(seed=111)
    step(60)
    s0 = float(scene.score()[0])
    # lifted-only: hold the load in the air over its spawn slot for a few substeps
    lp = block_local("load")
    for _ in range(8):
        st = torch.zeros(1, 13, device=device)
        st[0, 0], st[0, 1], st[0, 2] = lp[0], lp[1], 0.30
        st[0, 3] = 1.0
        st[0, 0:3] += scene.env_origins[0]
        scene.blocks["load"].write_root_state_to_sim(st, ids)
        step(1)
    s1 = float(scene.score()[0])
    put_block("load", (float(lp[0]), float(lp[1])), on="ground", settle=120)  # set it down
    put_block("foam_0", (0.0, 0.0), settle=240)  # foam partial on the plate
    s2 = float(scene.score()[0])
    print(f"[smoke] staged partials: reset={s0:.2f} lifted={s1:.2f} "
          f"foam-on-plate={s2:.2f}", flush=True)
    check("staged partials monotone: 0 <= reset < lifted < foam-on-plate < 1.0",
          s0 <= 0.005 and 0.15 <= s1 < s2 < 1.0)

    # =========================== save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.weighbridge")
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

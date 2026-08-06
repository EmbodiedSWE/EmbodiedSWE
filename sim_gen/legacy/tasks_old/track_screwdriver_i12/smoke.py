"""Smoke / oracle test for ToolHangupScene — NullRobot, teleport-oracle, RECORDED.

Battery (linear run, tee_up/pen_holder smoke skeleton):
  1. settle/no-NaN       — reset layout settles finite, tools lying on the floor, score 0;
  2. randomization       — seeded resets, READBACK stand/tool poses differ AND the keyed
                           ring order along the row permutes across episodes;
  3. subset-is-real      — present-tool count varies (2 vs 3) across seeded resets;
  4. null-policy         — 240 idle steps, score stays ~0, no success;
  5. oracle (3 seeds)    — kinematic carry per PRESENT tool: lift off the floor, reorient
                           vertical, align over its MATCHING ring, descend threading the
                           shaft through the aperture, RELEASE 15 mm high — the hang is a
                           genuine drop-and-catch; success() on every seed, rubric monotone
                           with a graded midpoint;
  6. negative controls   — (a) the SEED'S OWN STRATEGY: carry the tool along a five-
                           waypoint aerial path and keep holding (track_screwdriver's whole
                           task) — score pinned at the tiny raise credit, no success, and
                           releasing there just drops the tool on the floor;
                           (b) tool laid ACROSS the top of its own ring — never hung;
                           (c) wrong ring, fall-through: the small tool dropped through the
                           LARGE ring falls to the floor (handle < aperture, honest keying);
                           (d) wrong ring, blocked: the large tool cannot enter the SMALL
                           ring (shaft > aperture);
                           (e) near-miss: threaded but held 45 mm above the rim — not hung
                           (z-band), and releasing it settles into a counted hang;
  7. calibration probe   — free-drop threading sweep (tip 20 mm above the ring, xy offset
                           0..30 mm, 3 seeds each): publishes thread rate vs offset — the
                           aperture funnel (aperture - shaft_r = 6 mm for the mid tool).

Records video (viewport rgb annotator, RTX driver-version override) during show +
oracle-seed-0 + fall-through phases and saves `frames.npz` in the CWD. Prints exactly
`SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes, then hard-exits (Kit teardown
hangs otherwise).

Run: python smoke.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math
import os
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robobench
from robobench.core import ENVS

robobench.discover()
try:  # forge runs `python -m simgen_tasks.<task>.smoke`; standalone runs `python smoke.py`
    from . import scene as task_scene  # noqa: F401
except ImportError:
    import scene as task_scene  # noqa: F401

# ----- module state wired up in main() ---------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        if rec and _REC["i"] % args.record_every == 0:
            for _f in range(3):  # flush accumulated history (ghosting fix)
                env.sim.render()
            arr = np.asarray(_REC["annot"].get_data())
            if arr.size:
                _REC["frames"].append(arr[..., :3].astype(np.uint8).copy())
        _REC["i"] += 1


def _refresh() -> None:
    _ENV.iscene.update(0.0)


def _all_ids():
    return torch.arange(_ENV.num_envs, device=_ENV.device)


def _write_tool(k: int, pos_w: torch.Tensor, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
    """Teleport tool k (world pos (N,3)), zero velocities."""
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    st[:, 3:7] = torch.tensor(quat, device=_ENV.device)
    _ENV.scene.tool_bodies[k].write_root_state_to_sim(st, _all_ids())
    _refresh()


def _carry_tool(k: int, targets: list[torch.Tensor], quat=(1.0, 0.0, 0.0, 0.0),
                speed: float = 0.008) -> None:
    """Kinematically carry tool k through straight-line waypoints ((N,3) world) at fixed
    orientation, rewriting its state before every physics step."""
    scene = _ENV.scene
    _refresh()
    cur = scene.tool_bodies[k].data.root_pos_w.clone()
    for tgt in targets:
        while True:
            delta = tgt - cur
            dist = delta.norm(dim=-1, keepdim=True)
            if float(dist.max()) < 1e-4:
                break
            cur = cur + delta / dist.clamp(min=1e-9) * dist.clamp(max=speed)
            _write_tool(k, cur, quat)
            _step(1)


def _settle_until(pred, max_steps: int = 420, poll: int = 15) -> bool:
    if pred():
        return True
    waited = 0
    while waited < max_steps:
        _step(poll)
        waited += poll
        if pred():
            return True
    return False


def _ring(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(ring axis xy_w (N,2), ring band TOP z_w (N,)) for stand k, from readback."""
    _refresh()
    sp = _ENV.scene.stands[k].data.root_pos_w
    return sp[:, :2].clone(), sp[:, 2] + _ENV.scene.cfg.ring_top_z


def _pt(xy: torch.Tensor, z) -> torch.Tensor:
    p = torch.zeros(_ENV.num_envs, 3, device=_ENV.device)
    p[:, 0:2] = xy
    p[:, 2] = z
    return p


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:16s} | present={scene.present[0].int().tolist()} "
          f"hung={scene.hung()[0].int().tolist()} counted={scene.counted()[0].int().tolist()} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _perm_signature() -> tuple:
    """Order of the three keyed stands along the rack row (readback), a permutation of
    (0,1,2). Row yaw is <= 25 deg, so sorting by world y is order-along-the-row."""
    _refresh()
    ys = [float(_ENV.scene.stands[k].data.root_pos_w[0, 1]) for k in range(3)]
    return tuple(int(i) for i in np.argsort(ys))


# ----- the exported oracle ---------------------------------------------------------------------
def oracle_solution(scene_or_env, on_stage=None) -> bool:
    """Teleport-oracle: for each PRESENT tool (largest first), lift it off the floor,
    reorient vertical (tip down), align over ITS matching ring, descend so the shaft
    threads through the aperture, and RELEASE with the handle 15 mm above the rim — the
    catch-and-hang is honest drop physics. Returns success() for env 0."""
    env = scene_or_env if hasattr(scene_or_env, "iscene") else scene_or_env.env
    scene = env.scene
    c = scene.cfg
    _refresh()
    first = True
    for k in (2, 1, 0):
        if not bool(scene.present[0, k]):
            continue
        shaft_l = c.tools[k][2]
        xy, ring_top = _ring(k)
        tp = scene.tool_bodies[k].data.root_pos_w.clone()
        cruise_z = float(ring_top[0]) + 0.12  # tip cruises above every ring top
        # lift vertical, cruise over the matching ring
        _carry_tool(k, [_pt(tp[:, :2], cruise_z), _pt(xy, cruise_z)])
        if on_stage and first:
            on_stage("raised")
            first = False
        # descend: the tip threads through the aperture; stop with the handle bottom
        # 15 mm above the rim, then release -> drop-and-catch
        thread_z = ring_top + 0.015 - shaft_l
        _carry_tool(k, [_pt(xy, thread_z)], speed=0.004)
        ok = _settle_until(lambda k=k: bool(scene.counted()[0, k]))
        if not ok:  # one retry: re-thread a touch deeper
            print(f"[smoke]   oracle retry: tool {k} did not catch", flush=True)
            _carry_tool(k, [_pt(xy, cruise_z), _pt(xy, ring_top + 0.008 - shaft_l)],
                        speed=0.004)
            ok = _settle_until(lambda k=k: bool(scene.counted()[0, k]))
        if on_stage:
            on_stage(f"hung-{c.tools[k][0]}")
    return bool(scene.success()[0])


# ----- main ------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("sim_gen.tool_hangup")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.05, -1.05, 0.70)) + o),
                                tuple(np.array((0.05, 0.0, 0.16)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (960, 600))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        _REC["annot"] = annot if warm.size else None
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def thread_pose(tool_k: int, stand_k: int, hb_above: float,
                    dx: float = 0.0, dy: float = 0.0) -> torch.Tensor:
        """Tip position (N,3) putting tool_k axis-vertical at stand_k's ring axis with its
        handle bottom `hb_above` above the rim (negative tip z offset = threaded)."""
        xy, ring_top = _ring(stand_k)
        p = _pt(xy, ring_top + hb_above - c.tools[tool_k][2])
        p[:, 0] += dx
        p[:, 1] += dy
        return p

    # ================= 1. settle / no-NaN ================================================
    scene.cfg.subset_sample = False  # deterministic full layout for the control phases
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("show")
    states = torch.cat([t.data.root_state_w for t in scene.tool_bodies]
                       + [s.data.root_state_w for s in scene.stands], dim=-1)
    pos, axis, _v = scene._tool_tensors()
    lying = bool((pos[0, :, 2] < 0.06).all()) and bool((axis[0, :, 2].abs() < 0.5).all())
    check("settle/no-NaN: layout settles finite, tools lying flat on the floor, score 0",
          bool(torch.isfinite(states).all()) and lying
          and float(scene.score()[0]) == 0.0)
    _REC["on"] = False

    # ================= 2. randomization is real (readback) ===============================
    torch.manual_seed(101)
    env.reset()
    a_stand = scene.stands[0].data.root_pos_w[0, :2].clone()
    a_tool = scene.tool_bodies[1].data.root_pos_w[0, :2].clone()
    torch.manual_seed(202)
    env.reset()
    b_stand = scene.stands[0].data.root_pos_w[0, :2].clone()
    b_tool = scene.tool_bodies[1].data.root_pos_w[0, :2].clone()
    d_stand = float((a_stand - b_stand).norm())
    d_tool = float((a_tool - b_tool).norm())
    print(f"[smoke] randomization deltas: stand0={d_stand * 1000:.1f}mm "
          f"tool_m={d_tool * 1000:.1f}mm", flush=True)
    check("randomization-is-real: stand/tool readback differs across resets",
          d_stand > 0.005 and d_tool > 0.005)

    sigs = set()
    for seed in range(300, 306):
        torch.manual_seed(seed)
        env.reset()
        sigs.add(_perm_signature())
    print(f"[smoke] ring-order signatures over 6 resets: {sorted(sigs)}", flush=True)
    check("randomization: keyed ring order along the row permutes across episodes",
          len(sigs) >= 2)

    # ================= 3. subset sampling is real ========================================
    scene.cfg.subset_sample = True
    counts = []
    for seed in range(400, 408):
        torch.manual_seed(seed)
        env.reset()
        counts.append(int(scene.present[0].sum()))
    print(f"[smoke] present counts over 8 resets: {counts}", flush=True)
    check("subset-is-real: present-tool count varies (both 2 and 3 occur)",
          2 in counts and 3 in counts)
    scene.cfg.subset_sample = False

    # ================= 4. null policy fails ==============================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 5. oracle on 3 seeds + rubric shape ===============================
    scene.cfg.subset_sample = True
    stage_scores: list[float] = []

    def on_stage(tag: str) -> None:
        _refresh()
        stage_scores.append(float(scene.score()[0]))
        _report(f"oracle-{tag}")

    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        env.reset()
        _step(30)
        _REC["on"] = seed == 0
        stage_scores.clear()
        stage_scores.append(float(scene.score()[0]))
        ok = oracle_solution(env, on_stage=on_stage if seed == 0 else None)
        _report(f"oracle-seed-{seed}")
        check(f"oracle reaches success() (seed {seed}, "
              f"{int(scene.present[0].sum())}/3 present)", ok)
        if seed == 0:
            # stage_scores = [reset, raised, hung-..., ..., (final = 1.0 on success)]
            check("rubric: raise latch gives a small nonzero credit before any hang",
                  0.0 < stage_scores[1] <= 0.06)
            check("rubric: first hang lands in the graded middle (0.2 < s < 0.95)",
                  0.2 < stage_scores[2] < 0.95)
            check("rubric: monotone 0 -> raised -> hangs -> 1.0",
                  stage_scores == sorted(stage_scores) and stage_scores[0] == 0.0
                  and stage_scores[-1] == 1.0)
        _REC["on"] = False
    scene.cfg.subset_sample = False

    # ================= 6a. negative: the seed's own strategy =============================
    # track_screwdriver's ENTIRE task: object in hand, follow five aerial waypoints, keep
    # holding. Expressed here as a kinematic carry of the mid tool along a seed-like path
    # (away from the rack), then a held hover — and then the release, which just drops the
    # tool on the floor. Neither state may approach success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    dev = env.device
    org = env.iscene.env_origins
    wps = [(-0.45, -0.28, 0.15), (-0.44, -0.19, 0.22), (-0.40, -0.11, 0.36),
           (-0.37, 0.01, 0.47), (-0.35, 0.13, 0.46)]
    targets = [org + torch.tensor(w, device=dev) for w in wps]
    _carry_tool(1, targets)
    for _ in range(60):  # hold at the final waypoint (the seed's terminal state)
        _write_tool(1, targets[-1])
        _step(1)
    _write_tool(1, targets[-1])
    _report("seed-track-held")
    check("negative (seed strategy, held): waypoint tracking pins score at the raise "
          "credit, no hang, no success",
          float(scene.score()[0]) <= 0.051 and int(scene.counted()[0].sum()) == 0
          and not bool(scene.success()[0]))
    _step(180)  # release: the tool free-falls to the floor
    _report("seed-track-drop")
    _refresh()
    check("negative (seed strategy, released): tool ends on the floor, still no hang",
          float(scene.tool_bodies[1].data.root_pos_w[0, 2]) < 0.10
          and int(scene.counted()[0].sum()) == 0 and float(scene.score()[0]) <= 0.051)

    # ================= 6b. negative: laid ACROSS the ring top ============================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    xy1, top1 = _ring(1)
    lh = c.tools[1][2] + c.tools[1][4]  # tool length
    p = _pt(xy1, top1 + c.tools[1][3] + 0.002)
    p[:, 0] -= lh / 2  # center the tool over the ring axis (root is the tip)
    q_flat = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # axis -> +x
    _write_tool(1, p, q_flat)
    check("negative (across-ring): tool lying on the rim is not hung (static)",
          not bool(scene.hung()[0, 1]))
    _step(150)
    _report("across-ring")
    check("negative (across-ring): still not counted after settling",
          not bool(scene.counted()[0, 1]))

    # ================= 6c. negative: wrong ring — small tool FALLS THROUGH ===============
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    _write_tool(0, thread_pose(0, 2, 0.015))  # small tool threaded into the LARGE ring
    _settle_until(lambda: bool(scene.settled()[0, 0])
                  and float(scene.tool_bodies[0].data.root_pos_w[0, 2]) < 0.10,
                  max_steps=300)
    _report("fall-through")
    _refresh()
    check("negative (wrong ring, honest keying): small tool falls THROUGH the large ring "
          "to the floor — nothing counted",
          float(scene.tool_bodies[0].data.root_pos_w[0, 2]) < 0.10
          and int(scene.counted()[0].sum()) == 0 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6d. negative: wrong ring — large shaft BLOCKED ====================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_tool(2, thread_pose(2, 0, c.tools[2][2] + 0.008))  # tip 8 mm above small ring
    _step(300)
    _report("blocked")
    check("negative (wrong ring, honest keying): large shaft cannot enter the small ring "
          "— never hangs",
          not bool(scene.hung()[0, 2]) and int(scene.counted()[0].sum()) == 0)

    # ================= 6e. near-miss: threaded but held above the rim ====================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _write_tool(1, thread_pose(1, 1, 0.045))  # threaded, handle 45 mm above the rim
    check("negative (near-miss): threaded but 45 mm above the rim is NOT hung (z-band)",
          not bool(scene.hung()[0, 1]))
    ok = _settle_until(lambda: bool(scene.counted()[0, 1]))
    _report("near-miss-drop")
    check("tolerance boundary is real: releasing that near-miss settles into a counted "
          "hang", ok)

    # ================= 7. calibration probe: threading funnel sweep ======================
    print("[smoke] CALIBRATION SWEEP (free-drop offset -> thread rate, mid tool, tip "
          "20 mm above its ring, 3 seeds each; funnel = aperture - shaft_r = "
          f"{(c.apertures[1] - c.tools[1][1]) * 1000:.0f} mm)", flush=True)
    rates: dict[float, int] = {}
    for off_mm in (0.0, 3.0, 6.0, 9.0, 12.0, 30.0):
        hits = 0
        for s_i in range(3):
            torch.manual_seed(500 + int(off_mm) * 10 + s_i)
            env.reset()
            _step(20)
            ang = 2 * math.pi * s_i / 3.0
            _write_tool(1, thread_pose(1, 1, c.tools[1][2] + 0.020,
                                       dx=off_mm / 1000.0 * math.cos(ang),
                                       dy=off_mm / 1000.0 * math.sin(ang)))
            hit = _settle_until(lambda: bool(scene.counted()[0, 1]), max_steps=300)
            hits += int(hit)
            print(f"[smoke]   off={off_mm:.0f}mm seed={s_i}: hung={hit}", flush=True)
        rates[off_mm] = hits
    print("[smoke] SWEEP RESULT: " +
          " | ".join(f"{k:.0f}mm: {v}/3" for k, v in rates.items()), flush=True)
    check("calibration: centered free drop threads >= 2/3", rates[0.0] >= 2)
    check("calibration: far off-funnel drop (30 mm) threads <= 1/3", rates[30.0] <= 1)

    # ================= save + verdict ====================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="sim_gen.tool_hangup")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

    # hard exit — Kit teardown hangs otherwise
    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

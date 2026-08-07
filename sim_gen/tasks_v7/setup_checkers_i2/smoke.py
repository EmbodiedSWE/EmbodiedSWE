"""Smoke / rubric-REJECTION battery for CheckerSiloScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the latched
credit is monotone along a real trajectory). This battery proves the rubric REJECTS
wrong outcomes, and that the geometric claims the task rests on — the one-disc channel
holds what it is given, the slit cannot pass a checker — are physically load-bearing.
Every probe is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution: no probe here reaches success().

Checks:
   1. settle/no-NaN      — seeded reset settles finite; all four checkers stand on
                           edge in the rack, none inside the silo; score 0, no success;
   2. randomization      — two seeded resets: READBACK silo yaw, silo xy, rack xy
                           all differ;
   3. slot permutation   — over 10 resets the rack's slot->color pattern takes >= 3
                           distinct values and slot 0 holds both colors at least once;
   4. null-policy        — 240 idle steps: score ~0, no success (checkers stay put);
   5. SLIT INTERLOCK     — a checker seated in the channel is shoved toward the front
                           slit at 3x its own weight for 1.5 s: it never leaves the
                           silo ("cannot be taken back out" is physics, not fiat);
   6. SEED STRATEGY      — the seed's plan ("place the checkers flat on the board"):
                           R, W, R CONSTRUCTED lying flat in a neat row on the ground
                           beside the silo -> NO success, score ~0;
   7. wrong place        — a correct-order R/W/R stack CONSTRUCTED as a flat pancake
                           pile on the ground beside the silo -> no success, score ~0;
   8. out-of-order       — stack CONSTRUCTED reading [W, R, R] inside the channel:
                           right discs, wrong sequence -> no prefix credit beyond
                           approach, no success;
   9. near-miss          — correct [R, W] prefix in the channel, the second red
                           standing against the silo's OUTSIDE wall (centimetres from
                           its goal) -> not inside, no success, score capped at 0.75;
  10. exactly-three      — all FOUR checkers in the channel, bottom-up [R, W, R, W]
                           (a correct prefix plus the forbidden spare) -> no success;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.setup_checkers_i2.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=12)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects RTX
# -> the annotator returns EMPTY frames. Disable the check.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()

try:
    from . import scene as task_scene
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene

_qmul, _qy = task_scene._qmul, task_scene._qy

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    k, _zs, red = scene.stack_readout()
    stack = "".join("R" if bool(red[0, j]) else "W" for j in range(int(k[0])))
    print(f"[smoke] {tag:18s} | stack=[{stack}] k={int(k[0])} "
          f"appr={bool(scene._appr[0])} p1={bool(scene._p1[0])} "
          f"p2={bool(scene._p2[0])} score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


def _push(body, force_w: torch.Tensor, steps: int) -> None:
    """Apply a world-frame force at the body's CoM for `steps` steps, then clear."""
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = force_w.view(1, 1, 3).expand(n, 1, 3).contiguous()
    for _ in range(steps):
        body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)
        _step(1)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.checker_silo")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.85, 0.75)) + o),
                                tuple(np.array((0.32, -0.06, 0.12)) + o),
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

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def dyaw(a: float, b: float) -> float:
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    def silo_world(loc_xyz) -> torch.Tensor:
        """Silo-local point -> world (per-env)."""
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.silo.data.root_pos_w + quat_apply(scene.silo.data.root_quat_w, loc)

    def edge_quat() -> torch.Tensor:
        """On-edge orientation: disc axis along the silo slit axis (local +x)."""
        _refresh()
        return _qmul(scene.silo.data.root_quat_w,
                     _qy(torch.full((n,), math.pi / 2, device=device)))

    def build_stack(names: list[str]) -> None:
        """CONSTRUCT `names` (bottom-up) seated in the channel, then settle. All
        writes land before the first step so no intermediate state is judged."""
        q = edge_quat()
        for j, nm in enumerate(names):
            z = c.z_floor + c.disc_r + j * (2 * c.disc_r + 0.001) + 0.002
            _write_body(scene.discs[nm], silo_world((0.0, 0.0, z)), q)
        _step(90)

    def rack_pattern() -> str:
        slot_of = scene.slot_of[0].tolist()
        col = {slot_of[i]: ("R" if scene.IS_RED[i] else "W") for i in range(4)}
        return "".join(col[s] for s in range(4))

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(90)
    _report("settle")
    _REC["on"] = False
    pos, _q, _v = scene._disc_tensors()
    hz = pos[0, :, 2] - env.iscene.env_origins[0, 2]
    check("settle/no-NaN: layout settles finite; four checkers on edge in the rack, "
          "none inside the silo; score 0, no success",
          bool(torch.isfinite(pos).all())
          and bool((hz > 0.015).all() and (hz < 0.06).all())
          and int(scene.in_silo()[0].sum()) == 0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene.silo.data.root_quat_w[0]),
                scene.silo.data.root_pos_w[0, :2].clone(),
                scene.rack.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_sp, a_rp = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_sp, b_rp = readback()
    d_yawv = dyaw(a_yaw, b_yaw)
    d_sp, d_rp = float((a_sp - b_sp).norm()), float((a_rp - b_rp).norm())
    print(f"[smoke] randomization deltas: silo_yaw={d_yawv:.1f}deg "
          f"silo_xy={d_sp * 1000:.1f}mm rack_xy={d_rp * 1000:.1f}mm", flush=True)
    check("randomization-is-real: silo yaw, silo xy, rack xy readback differ",
          d_yawv > 2.0 and d_sp > 0.003 and d_rp > 0.003)

    # ================= 3. slot permutation coverage ===============================================
    patterns, slot0 = set(), set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        p = rack_pattern()
        patterns.add(p)
        slot0.add(p[0])
    print(f"[smoke] over 10 resets: rack patterns {sorted(patterns)}", flush=True)
    check("slot permutation: >= 3 distinct rack color patterns over 10 resets and "
          "slot 0 holds both colors",
          len(patterns) >= 3 and slot0 == {"R", "W"})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. SLIT INTERLOCK: a seated checker cannot be pushed back out ==============
    # "a checker that has dropped in cannot be taken back out": shove a seated disc
    # toward the front slit at 3x its weight for 1.5 s — the 24 mm slit vs the 40 mm
    # disc must hold it. This is the physics behind the irreversibility claim.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    build_stack(["red_0"])
    _refresh()
    assert bool(scene.in_silo()[0, 0]), "probe setup: red_0 must seat in the channel"
    slit_w = quat_apply(scene.silo.data.root_quat_w,
                        torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]
    f = 3.0 * c.disc_mass * 9.81 * slit_w
    x_max = -1.0
    for _ in range(9):  # 9 x 20 = 180 steps = 1.5 s of shoving
        _push(scene.discs["red_0"], f, 20)
        _refresh()
        x_max = max(x_max, float(
            scene._silo_local(scene.discs["red_0"].data.root_pos_w)[0, 0]))
    _step(60)
    _report("slit-interlock")
    print(f"[smoke] slit shove: max silo-frame x={x_max * 1000:.1f}mm "
          f"(slit half-width {c.slit_w / 2 * 1000:.0f}mm, disc r "
          f"{c.disc_r * 1000:.0f}mm)", flush=True)
    check("SLIT INTERLOCK: 3x-weight shove toward the slit for 1.5 s never gets the "
          "seated checker out of the silo",
          bool(scene.in_silo()[0, 0]) and x_max < c.in_x_tol
          and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. negative: the SEED'S OWN STRATEGY =======================================
    # "Place the checkers on the board in the starting arrangement" — CONSTRUCT the
    # seed's kind of end state: R, W, R laid FLAT in a neat row on the open ground
    # beside the silo (flat placement on a horizontal surface, the only skill the
    # seed's plan exercises). Nothing entered the silo -> must NOT be success.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    for j, nm in enumerate(["red_0", "white_0", "red_1"]):
        p = silo_world((0.12 + 0.05 * j, -0.10, 0.0))
        p[:, 2] = env.iscene.env_origins[:, 2] + c.disc_h / 2 + 0.002
        _write_body(scene.discs[nm], p, None)  # identity quat = lying flat
    _step(120)
    _report("seed-strategy")
    check("negative (SEED strategy): R/W/R laid flat in a row on the ground beside "
          "the silo — nothing inside, NO success, score ~0",
          int(scene.in_silo()[0].sum()) == 0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)
    _REC["on"] = False

    # ================= 7. negative: right order, wrong place ======================================
    # A pancake pile reading R/W/R bottom-up — the CORRECT color sequence — built on
    # the open ground beside the silo. Order alone is not the task; the silo is.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    base = silo_world((0.13, -0.11, 0.0))
    for j, nm in enumerate(["red_0", "white_0", "red_1"]):
        p = base.clone()
        p[:, 2] = env.iscene.env_origins[:, 2] + c.disc_h / 2 + 0.002 \
            + j * (c.disc_h + 0.001)
        _write_body(scene.discs[nm], p, None)
    _step(120)
    _report("wrong-place")
    check("negative (wrong place): correct-order R/W/R pancake stack on the ground "
          "beside the silo — no success, score ~0",
          int(scene.in_silo()[0].sum()) == 0 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.05)

    # ================= 8. negative: out-of-order stack ============================================
    # The channel loaded [W, R, R] bottom-up: same three discs, wrong sequence. The
    # prefix credit and success must both refuse (order is the task).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    build_stack(["white_0", "red_0", "red_1"])
    _report("out-of-order")
    k, _zs, red = scene.stack_readout()
    seq = "".join("R" if bool(red[0, j]) else "W" for j in range(int(k[0])))
    check("negative (out-of-order): channel stack reads [WRR] — no prefix credit "
          "beyond approach (p1/p2 unlatched), no success, score <= 0.16",
          seq == "WRR" and not bool(scene._p1[0]) and not bool(scene._p2[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.16)
    _REC["on"] = False

    # ================= 9. near-miss: third red centimetres away, OUTSIDE the wall =================
    # Correct [R, W] prefix in the channel; the second red stands on the silo's foot
    # plate pressed against the OUTSIDE of the wall — as close as it can be without
    # being in. k stays 2: no success, capped partial credit.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    build_stack(["red_0", "white_0"])
    p = silo_world((0.0, c.chan_y / 2 + c.wall_t + c.disc_h / 2 + 0.004,
                    0.012 + c.disc_r + 0.002))
    _write_body(scene.discs["red_1"], p, edge_quat())
    _step(120)
    _report("near-miss")
    check("near-miss: [R,W] correct in the channel, second red standing against the "
          "silo's OUTSIDE wall — not inside, no success, score <= 0.75",
          not bool(scene.in_silo()[0, 1]) and int(scene.stack_readout()[0][0]) == 2
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.75)

    # ================= 10. negative: exactly-three violated =======================================
    # All FOUR checkers in the channel, bottom-up [R, W, R, W]: the target pattern
    # IS a prefix of this stack, but the forbidden spare white is in. Count gate
    # must refuse. (Constructed in one batch: k == 4 from the first judged frame,
    # so no intermediate state is ever a success.)
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    build_stack(["red_0", "white_0", "red_1", "white_1"])
    _report("four-in")
    check("negative (exactly-three): all four checkers stacked [R,W,R,W] in the "
          "channel — count gate refuses, no success, score <= 0.75",
          int(scene.stack_readout()[0][0]) == 4 and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.75)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.checker_silo")
        print(f"[smoke] saved {arr.shape} -> {args.out}", flush=True)
    check("video: frames captured and saved to frames.npz", len(_REC["frames"]) > 10)

    n_pass = sum(ok for _n, ok in checks)
    n_tot = len(checks)
    if n_pass == n_tot:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{n_tot}", flush=True)
        code = 0
    else:
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{n_tot}", flush=True)
        code = 1

    threading.Timer(10.0, lambda: os._exit(code)).start()
    try:
        env.close()
        app.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


if __name__ == "__main__":
    main()

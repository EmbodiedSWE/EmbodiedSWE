"""Smoke / rubric-REJECTION battery for CrateTurnoverScene — NullRobot, teleported probes.

solve.py (the real Franka solution) is the acceptance proof: it demonstrates the rubric
accepts correct outcomes and that latched credit is monotone along a real trajectory.
This battery proves the rubric REJECTS wrong outcomes. Every probe is CONSTRUCTED as a
settled state (teleport, real physics steps, judge) — instrumentation, never a solution:
no probe here reaches success().

Checks:
   1. settle/no-NaN     — reset layout settles finite, milk starts IN the crate, score 0;
   2. randomization     — two seeded resets: READBACK crate pos+yaw, pad, milk, lid differ;
   3. subset sampling   — present-count varies across 10 resets (1 and 2 both occur);
   4. null-policy       — 240 idle steps: score ~0, no success (the seed's terminal
                          relation, milk-in-basket, is the START state and earns nothing);
   5. SEED STRATEGY     — the seed's plan taken further (milk kept in the crate, all other
                          courtesies granted: items stowed, lid laid on): the lid rests
                          HIGH on the 140 mm carton and never seats — no success, <= 0.35;
   6. near-miss (pad)   — milk upright but 110 mm past the pad center: no success;
   7. near-miss (flat)  — milk lying on its side centered on the pad: no success;
   8. near-miss (lid xy)— lid released 45 mm off-axis topples into the mouth: never closed;
   9. near-miss (yaw)   — lid resting at the exact seat pose twisted 30 deg: rejected;
  10. wrong place       — items resting ON TOP of a seated lid earn no stow credit;
  11. incomplete        — one item left on the floor, everything else right: no success;
12-13. calibration      — lid-release xy-offset sweep on the (milk-evicted, still
                          unloaded) crate: <= 10 mm offsets seat reliably (the 15 mm gate
                          is physically reachable), >= 25 mm never judged seated (25 mm
                          rests level but outside the gate; 45 mm topples in);
  14. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_pick_milk_i3.smoke --headless
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
    from . import scene as task_scene  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import scene as task_scene  # noqa: F401

# ----- module state wired up in main() ----------------------------------------------------------
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


def _settle_until(pred, max_steps: int = 300, poll: int = 15) -> bool:
    _step(poll)  # ALWAYS force real steps first (zero-step teleport trap)
    if pred():
        return True
    waited = poll
    while waited < max_steps:
        _step(poll)
        waited += poll
        if pred():
            return True
    return False


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:18s} | milk_in_crate={bool(scene.milk_in_crate()[0])} "
          f"delivered={bool(scene.delivered()[0])} "
          f"stowed={scene.stowed()[0].int().tolist()} "
          f"present={scene.present[0].int().tolist()} "
          f"lid_seated={bool(scene.lid_seated()[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _quats():
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
    return quat_apply, quat_apply_inverse, quat_mul


def _crate_pose():
    scene = _ENV.scene
    _refresh()
    return scene.crate.data.root_pos_w.clone(), scene.crate.data.root_quat_w.clone()


def _seat_pose():
    """(pos (N,3), quat (N,4)) of the flush-seated lid (board center) in world frame."""
    scene = _ENV.scene
    cp, cq = _crate_pose()
    p = cp.clone()
    p[:, 2] += scene.cfg.seat_z_local
    return p, cq


def _pad_top_pose():
    scene = _ENV.scene
    _refresh()
    pp = scene.pad.data.root_pos_w.clone()
    pp[:, 2] += scene.cfg.pad_t / 2
    return pp


def _stow_target(i: int):
    """(near-rest pos (N,3), quat (N,4)) for item i inside the crate (crate-frame slot)."""
    quat_apply, _qai, _qm = _quats()
    scene = _ENV.scene
    c = scene.cfg
    cp, cq = _crate_pose()
    n = _ENV.num_envs
    sx, sy = c.stow_slots[i]
    _name, kind, r, height, _m, _rgb = c.items[i]
    half = height / 2 if kind == "cyl" else r
    loc = torch.zeros(n, 3, device=_ENV.device)
    loc[:, 0], loc[:, 1] = sx, sy
    loc[:, 2] = c.bot_t + half + 0.004
    return cp + quat_apply(cq, loc), cq.clone()


def _deliver_milk_teleport(offset_xy=(0.0, 0.0), lying: bool = False) -> None:
    """Construct a (near-)delivered milk pose and settle (probe constructor)."""
    scene = _ENV.scene
    c = scene.cfg
    pt = _pad_top_pose()
    p = pt.clone()
    p[:, 0] += offset_xy[0]
    p[:, 1] += offset_xy[1]
    if lying:
        p[:, 2] += c.milk_w / 2 + 0.004
        q = torch.zeros(_ENV.num_envs, 4, device=_ENV.device)
        c45 = math.cos(math.pi / 4)
        q[:, 0], q[:, 2] = c45, c45  # pitched 90 deg: lying on a side face
        _write_body(scene.milk, p, q)
    else:
        p[:, 2] += c.milk_h / 2 + 0.004
        _write_body(scene.milk, p)
    _step(60)


def _park_milk_on_floor() -> None:
    """Milk out of the crate to a neutral floor spot (NOT the pad — not delivered)."""
    scene = _ENV.scene
    c = scene.cfg
    p = torch.zeros(_ENV.num_envs, 3, device=_ENV.device)
    p[:, 0] = _ENV.iscene.env_origins[:, 0] - 0.15
    p[:, 1] = _ENV.iscene.env_origins[:, 1] + 0.05
    p[:, 2] = c.milk_h / 2 + 0.003
    _write_body(scene.milk, p)
    _step(30)


def _stow_items_teleport(indices) -> None:
    scene = _ENV.scene
    for i in indices:
        name = scene.cfg.items[i][0]
        tgt, tq = _stow_target(i)
        _write_body(scene.items[name], tgt, tq)
    _step(60)


def _lid_release_teleport(dz: float = 0.008, off_x: float = 0.0,
                          yaw_deg: float = 0.0) -> None:
    _q, _qai, quat_mul = _quats()
    seat_p, seat_q = _seat_pose()
    p = seat_p.clone()
    p[:, 0] += off_x
    p[:, 2] += dz
    if yaw_deg:
        half = math.radians(yaw_deg) / 2
        qz = torch.zeros(_ENV.num_envs, 4, device=_ENV.device)
        qz[:, 0], qz[:, 3] = math.cos(half), math.sin(half)
        seat_q = quat_mul(seat_q, qz)
    _write_body(_ENV.scene.lid, p, seat_q)
    _step(60)


# ----- main --------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("sim_gen.crate_turnover")().build(
        num_envs=args.num_envs, device=device,
        scene_cfg=task_scene.CrateTurnoverSceneCfg(subset_sample=False))
    _ENV = env
    scene = env.scene
    c = scene.cfg
    quat_apply, quat_apply_inverse, quat_mul = _quats()

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.90, -0.80, 0.70)) + o),
                                tuple(np.array((0.06, -0.04, 0.06)) + o),
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

    # ================= 1. settle / no-NaN =====================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("show")
    _REC["on"] = False
    states = torch.cat([scene.milk.data.root_state_w, scene.lid.data.root_state_w,
                        scene.crate.data.root_state_w, scene.pad.data.root_state_w]
                       + [b.data.root_state_w for b in scene.items.values()], dim=-1)
    check("settle/no-NaN: layout settles finite, milk starts IN the crate, score 0",
          bool(torch.isfinite(states).all()) and bool(scene.milk_in_crate()[0])
          and float(scene.score()[0]) == 0.0 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ====================================
    def readback():
        _refresh()
        return (scene.crate.data.root_pos_w[0, :2].clone(),
                yaw_of(scene.crate.data.root_quat_w[0]),
                scene.pad.data.root_pos_w[0, :2].clone(),
                scene.milk.data.root_pos_w[0, :2].clone(),
                scene.lid.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    a_cp, a_cy, a_pp, a_mp, a_lp = readback()
    torch.manual_seed(202)
    env.reset()
    b_cp, b_cy, b_pp, b_mp, b_lp = readback()
    d_c, d_p = float((a_cp - b_cp).norm()), float((a_pp - b_pp).norm())
    d_m, d_l = float((a_mp - b_mp).norm()), float((a_lp - b_lp).norm())
    d_cy = dyaw(a_cy, b_cy)
    print(f"[smoke] randomization deltas: crate={d_c * 1000:.1f}mm/{d_cy:.1f}deg "
          f"pad={d_p * 1000:.1f}mm milk={d_m * 1000:.1f}mm lid={d_l * 1000:.1f}mm",
          flush=True)
    check("randomization-is-real: crate pos+yaw, pad, milk, lid readback differ",
          d_c > 0.005 and d_cy > 3.0 and d_p > 0.005 and d_m > 0.005 and d_l > 0.005)

    # ================= 3. subset sampling is real ==============================================
    scene.cfg.subset_sample = True
    counts = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        counts.add(int(scene.present[0].sum()))
    scene.cfg.subset_sample = False
    print(f"[smoke] present-count values over 10 resets: {sorted(counts)}", flush=True)
    check("subset sampling: present count varies across resets (1 and 2 both occur)",
          counts == {1, 2})

    # ================= 4. null policy fails ====================================================
    # The seed's terminal relation (milk in the basket) is this task's START state: doing
    # nothing IS the seed outcome, and it must earn ~0.
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps (milk stays in the crate), score ~0, "
          "no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY, taken further ====================
    # Milk kept in the crate; every other courtesy granted (items stowed, lid laid on).
    # The lid rests HIGH on the 140 mm carton and never seats; score stays <= 0.35.
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _REC["on"] = True
    # milk to a crate corner (crate frame) so the items fit beside it without penetration
    cp, cq = _crate_pose()
    loc = torch.zeros(env.num_envs, 3, device=env.device)
    loc[:, 0], loc[:, 1] = -0.030, -0.030
    loc[:, 2] = c.milk_rest_z + 0.002
    _write_body(scene.milk, cp + quat_apply(cq, loc))
    _step(30)
    for i, (sx, sy) in enumerate(((0.040, 0.040), (0.040, -0.035))):
        name, kind, r, height, _m, _rgb = c.items[i]
        half = height / 2 if kind == "cyl" else r
        loc = torch.zeros(env.num_envs, 3, device=env.device)
        loc[:, 0], loc[:, 1] = sx, sy
        loc[:, 2] = c.bot_t + half + 0.006
        _write_body(scene.items[name], cp + quat_apply(cq, loc), cq.clone())
    _step(60)
    # lid released above the protruding carton top — what "closing" costs here
    seat_p, seat_q = _seat_pose()
    drop = seat_p.clone()
    drop[:, 2] = cp[:, 2] + c.bot_t + c.milk_h + c.lid_t / 2 + 0.006
    _write_body(scene.lid, drop, seat_q)
    _step(240)
    _report("seed-strategy")
    check("negative (SEED strategy): milk kept in the crate — lid rests high on the "
          "carton, never seats: no success, score <= 0.35",
          bool(scene.milk_in_crate()[0]) and not bool(scene.lid_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.35)
    _REC["on"] = False

    # ================= 6. near-miss: milk upright but off the pad ==============================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _stow_items_teleport(range(len(c.items)))
    _lid_release_teleport()
    _deliver_milk_teleport(offset_xy=(0.0, 0.110))  # 110 mm off pad center: on the floor
    _report("milk-off-pad")
    check("negative (near-miss pad): milk upright 110 mm past the pad center — no "
          "delivery credit, no success",
          not bool(scene.delivered()[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.56)

    # ================= 7. near-miss: milk lying flat ON the pad ================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _stow_items_teleport(range(len(c.items)))
    _lid_release_teleport()
    _deliver_milk_teleport(lying=True)
    _report("milk-lying")
    check("negative (near-miss upright): milk lying on its side ON the pad — rejected "
          "by the upright gate, no success",
          not bool(scene.delivered()[0]) and not bool(scene.success()[0]))

    # ================= 8. near-miss: lid released 45 mm off-axis ===============================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _park_milk_on_floor()
    _stow_items_teleport(range(len(c.items)))
    _lid_release_teleport(off_x=0.045)
    _step(180)
    _report("lid-45mm-off")
    check("negative (near-miss lid xy): lid released 45 mm off-axis never judged closed",
          not bool(scene.lid_seated()[0]) and not bool(scene.success()[0]))

    # ================= 9. near-miss: lid twisted 30 deg at the exact seat ======================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _park_milk_on_floor()
    _stow_items_teleport(range(len(c.items)))
    _lid_release_teleport(dz=0.003, yaw_deg=30.0)
    _step(120)
    _report("lid-30deg")
    check("negative (near-miss lid yaw): lid resting on the rim twisted 30 deg is "
          "rejected by the mod-90 yaw gate",
          not bool(scene.lid_seated()[0]) and not bool(scene.success()[0]))

    # ================= 10. wrong place: items ON TOP of the seated lid =========================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _park_milk_on_floor()
    _lid_release_teleport()
    seat_p, seat_q = _seat_pose()
    for i, off in enumerate(((0.055, 0.0), (-0.055, 0.0))):
        name, kind, r, height, _m, _rgb = c.items[i]
        half = height / 2 if kind == "cyl" else r
        p = seat_p.clone()
        p[:, 0] += off[0]
        p[:, 1] += off[1]
        p[:, 2] += c.lid_t / 2 + half + 0.006
        _write_body(scene.items[name], p, seat_q.clone())
    _step(120)
    _report("items-on-lid")
    check("negative (wrong place): items resting ON the seated lid earn no stow credit, "
          "no success",
          int(scene.stowed()[0].sum()) == 0 and bool(scene.lid_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.30)

    # ================= 11. incomplete: one item left outside ===================================
    torch.manual_seed(100)
    env.reset()
    _step(40)
    _deliver_milk_teleport()
    _stow_items_teleport([0])  # cube in; can left at its floor slot
    _lid_release_teleport()
    _step(60)
    _report("incomplete")
    check("negative (incomplete): one item left on the floor, all else right — "
          "no success, score <= 0.71",
          bool(scene.delivered()[0]) and bool(scene.lid_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.71)

    # ================= 12-13. calibration: lid-release offset sweep ============================
    # Run on a milk-evicted but UNLOADED crate (items left at their floor slots) so no
    # sweep state is anywhere near success — this calibrates the lid gate in isolation.
    print("[smoke] CALIBRATION SWEEP (lid release offset along crate-x -> seated rate, "
          "3 seeds each; crate unloaded, milk parked off-pad)", flush=True)
    offsets = (0.0, 0.010, 0.025, 0.045)
    rates: dict[float, int] = {}
    for off_m in offsets:
        hits = 0
        for seed in (10, 11, 12):
            torch.manual_seed(seed)
            env.reset()
            _step(20)
            _park_milk_on_floor()
            _lid_release_teleport(off_x=off_m if seed % 2 == 0 else -off_m)
            hit = _settle_until(
                lambda: bool(scene.lid_seated()[0]) and bool(scene.lid_still()[0]),
                max_steps=240)
            hits += int(hit)
            print(f"[smoke]   off={off_m * 1000:.0f}mm seed={seed}: seated={hit}",
                  flush=True)
        rates[off_m] = hits
    small = sum(v for k, v in rates.items() if k <= 0.010)
    large = sum(v for k, v in rates.items() if k >= 0.025)
    print("[smoke] SWEEP RESULT: " +
          " | ".join(f"{k * 1000:.0f}mm: {v}/3" for k, v in rates.items()) +
          f"  -> small(<=10mm)={small}/6, large(>=25mm)={large}/6", flush=True)
    check("calibration: small offsets (<=10 mm) seat >= 5/6 (15 mm gate reachable)",
          small >= 5)
    check("calibration: large offsets (>=25 mm) seat <= 1/6", large <= 1)

    # ================= save + verdict ==========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="sim_gen.crate_turnover")
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

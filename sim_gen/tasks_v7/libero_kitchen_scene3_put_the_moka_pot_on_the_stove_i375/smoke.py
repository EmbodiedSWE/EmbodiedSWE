"""Smoke / rubric-REJECTION battery for FlameKeeperScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts the pot-first swap and the
latched credit is monotone along it). This battery proves the rubric REJECTS wrong
outcomes and that the claims the task rests on — the dead-man plate as a live,
physical order-enforcer, the permanence of a flame-out, and the placement clauses —
are load-bearing. Every probe is CONSTRUCTED as a settled state (teleport transport,
real physics steps, judge) — instrumentation, never a solution: no probe here
reaches success().

Checks:
  1.  settle/no-NaN    — seeded reset settles finite: pan pressing the plate, foul
                         armed, flame alive, score 0, no success;
  2.  randomization    — readback over seeds: pot xy AND pan yaw differ (3-seed
                         max-pairwise); BOTH plate halves occur as the pan's start
                         side over 6 seeds;
  3.  null-policy      — 300 idle steps: score ~0, no success;
  4.  FLAGSHIP illegal order — the SEED's plan: pan removed to the counter FIRST.
                         The unloaded plate PHYSICALLY pops past the foul band
                         (readback of the mechanism's response), the flame dies;
                         the pot is then installed on the pressed plate anyway. The
                         END STATE passes every geometric success clause and is
                         IDENTICAL to the winning arrangement — yet success is
                         refused and score == 0. 240 further steps: the foul never
                         clears (irreversible);
  5.  grace period     — the pan lifted off for a few steps and put straight back:
                         the plate never reaches the foul band, the flame survives
                         (the foul is not hair-triggered);
  6.  rim perch        — pot stood on the well ring beside the plate (pan still
                         pressing): near latch only, no on-plate credit, no
                         success, score <= 0.105;
  7.  swap incomplete  — pot correctly on the free half but the pan LEFT on the
                         plate: 0.35 latched, no swap credit, no success;
  8.  tipped pot       — pot lying on its SIDE on the free half: the upright/z
                         clauses refuse the on-plate latch, score <= 0.105;
  9.  pan too close    — legal order, but the pan set down inside the clearance
                         radius: no swap latch, no success, score <= 0.355;
  10. frames.npz       — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene3_put_the_moka_pot_on_the_stove_i375.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=24)
parser.add_argument("--out", type=str, default="frames.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe: kit mis-decodes the driver version on some GPUs and silently rejects
# RTX -> the annotator returns EMPTY frames. Disable the check.
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

_qz = task_scene._qz

# Global watchdog: if anything wedges, die loudly before the forge timeout.
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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


def _plate_hover(local_x: float, local_y: float = 0.0, dz: float = 0.010) -> torch.Tensor:
    """World hover point `dz` above the plate TOP at plate-local xy (LIVE pose —
    if the plate is popped up, the hover rides up with it)."""
    scene = _ENV.scene
    c = scene.cfg
    pos = scene.plate.data.root_pos_w.clone()
    pos[:, 0] += local_x
    pos[:, 1] += local_y
    pos[:, 2] += c.plate_size[2] + dz
    return pos


def _drop_pot_on_plate(local_x: float, yaw: float, settle: int = 480) -> bool:
    scene = _ENV.scene
    n = _ENV.num_envs
    q = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                     device=_ENV.device).expand(n, 4)
    for _ in range(3):
        _write_body(scene.pot, _plate_hover(local_x), q)
        _step(settle)
        if bool(scene.pot_on_plate()[0]) and bool(scene.pot_upright()[0]):
            return True
    return False


def _deck_spot(x: float, y: float, dz: float = 0.006) -> torch.Tensor:
    n = _ENV.num_envs
    c = _ENV.scene.cfg
    pos = torch.zeros(n, 3, device=_ENV.device)
    pos[:, 0], pos[:, 1] = x, y
    pos[:, 2] = c.deck_top + dz
    pos[:, :3] += _ENV.iscene.env_origins
    return pos


def _pan_side() -> float:
    scene = _ENV.scene
    return 1.0 if float(scene.pan.data.root_pos_w[0, 0]
                        - scene.plate.data.root_pos_w[0, 0]) > 0 else -1.0


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:16s} | q={float(scene.plate_q()[0]) * 1000:+6.2f}mm "
          f"pressed={bool(scene.plate_pressed()[0])} armed={bool(scene._armed[0])} "
          f"foul={bool(scene._foul[0])} pot_on={bool(scene.pot_on_plate()[0])} "
          f"pan_on={bool(scene.pan_on_plate()[0])} pan_clear={bool(scene.pan_clear()[0])} "
          f"latches=({int(scene._l_near[0])},{int(scene._l_on[0])},{int(scene._l_swap[0])}) "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.flame_keeper")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.20, -0.90, 0.80)) + o),
                                tuple(np.array((0.38, 0.02, 0.08)) + o),
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

    def pot_slot_yaw(side: float) -> float:
        return math.pi if side > 0 else 0.0

    oxy = env.iscene.env_origins[:, :2]

    # ================= 1. settle / no-NaN =======================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    fin = bool(torch.isfinite(scene.pot.data.root_pos_w).all()
               and torch.isfinite(scene.pan.data.root_pos_w).all()
               and torch.isfinite(scene.plate.data.root_pos_w).all()
               and torch.isfinite(scene.pot.data.root_lin_vel_w).all())
    check("settle/no-NaN: pan seated pressing the plate, foul armed, flame alive, "
          "score 0, no success",
          fin and bool(scene.pan_on_plate()[0]) and bool(scene.plate_pressed()[0])
          and bool(scene._armed[0]) and not bool(scene._foul[0])
          and float(scene.score()[0]) <= 0.005 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ======================================
    def readback():
        _refresh()
        return (scene.pot.data.root_pos_w[0, :2] - oxy[0],
                yaw_of(scene.pan.data.root_quat_w[0]))

    deltas_p, deltas_y = [], []
    obs = []
    for s in (101, 202, 303):
        torch.manual_seed(s)
        env.reset()
        _step(5)
        obs.append(readback())
    for i in range(3):
        for j in range(i + 1, 3):
            deltas_p.append(float((obs[i][0] - obs[j][0]).norm()))
            deltas_y.append(dyaw(obs[i][1], obs[j][1]))
    sides = set()
    for s in range(400, 406):
        torch.manual_seed(s)
        env.reset()
        _step(3)
        sides.add(_pan_side())
    print(f"[smoke] randomization: pot_xy max-pairwise={max(deltas_p) * 1000:.1f}mm "
          f"pan_yaw max-pairwise={max(deltas_y):.1f}deg sides={sorted(sides)}", flush=True)
    check("randomization-is-real: pot xy and pan yaw readback differ (3-seed "
          "max-pairwise); both plate halves occur over 6 seeds",
          max(deltas_p) > 0.005 and max(deltas_y) > 3.0 and len(sides) == 2)

    # ================= 3. null policy fails =====================================================
    torch.manual_seed(100)
    env.reset()
    _step(300)
    _report("null-policy")
    check("null-policy-fails: 300 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.02 and not bool(scene.success()[0]))

    # ================= 4. FLAGSHIP: the seed's plan (pan off FIRST) fouls forever ===============
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    side = _pan_side()
    assert bool(scene._armed[0]) and not bool(scene._foul[0])
    _write_body(scene.pan, _deck_spot(0.62, 0.18))  # the pan leaves the plate FIRST
    q_max = 0.0
    for _ in range(120):
        _step(1)
        q_max = max(q_max, float(scene.plate_q()[0]))
    _report("pan-first")
    popped = q_max > c.foul_q
    fouled = bool(scene._foul[0])
    print(f"[smoke] unloaded plate popped to q_max={q_max * 1000:.1f}mm "
          f"(foul band {c.foul_q * 1000:.1f}mm) -> foul={fouled}", flush=True)
    # ... and the pot is installed on the (now pressed-again) plate anyway:
    ok_drop = _drop_pot_on_plate(-side * c.pot_slot_x, pot_slot_yaw(side))
    _step(120)
    _report("illegal-final")
    geom = (bool(scene.pot_on_plate()[0]) and bool(scene.pot_upright()[0])
            and not bool(scene.pan_on_plate()[0]) and bool(scene.pan_clear()[0])
            and bool(scene.plate_pressed()[0]) and bool(scene.settled()[0]))
    refused = not bool(scene.success()[0]) and float(scene.score()[0]) <= 1e-6
    _step(240)
    _REC["on"] = False
    check("FLAGSHIP illegal order: pan removed first -> plate physically pops past "
          "the foul band, flame dies; end state passes EVERY geometric clause yet "
          "success refused, score == 0, and 240 steps later the foul still holds",
          popped and fouled and ok_drop and geom and refused
          and bool(scene._foul[0]) and float(scene.score()[0]) <= 1e-6
          and not bool(scene.success()[0]))

    # ================= 5. grace period: a brief lift does not foul ==============================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    pan_pos = scene.pan.data.root_pos_w.clone()
    pan_quat = scene.pan.data.root_quat_w.clone()
    up = pan_pos.clone()
    up[:, 2] += 0.15
    _write_body(scene.pan, up, pan_quat)  # pan lifted OFF
    q_max = 0.0
    for _ in range(5):
        _step(1)
        q_max = max(q_max, float(scene.plate_q()[0]))
    back = pan_pos.clone()
    back[:, 2] += 0.002
    _write_body(scene.pan, back, pan_quat)  # ... and put straight back
    _step(300)
    _report("grace")
    print(f"[smoke] grace lift: q_max={q_max * 1000:.1f}mm "
          f"(foul band {c.foul_q * 1000:.1f}mm)", flush=True)
    check("grace period: pan lifted for 5 steps and returned — the plate never "
          "reaches the foul band, the flame survives, plate pressed again",
          q_max < c.foul_q and not bool(scene._foul[0])
          and bool(scene.plate_pressed()[0]) and bool(scene.pan_on_plate()[0])
          and not bool(scene.success()[0]))

    # ================= 6. rim perch: pot on the well ring is NOT on the plate ===================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    # y ring wall center: housing-local y ~ (well_inner/2 + housing_outer/2)/2
    ry = (c.well_inner[1] / 2 + c.housing_outer[1] / 2) / 2
    perch = _deck_spot(c.housing_pos[0], c.housing_pos[1] + ry,
                       dz=c.housing_base_h + c.ring_h + 0.002)
    _write_body(scene.pot, perch)  # identity attitude
    _step(360)
    _report("rim-perch")
    check("rim perch: pot stood on the well ring beside the plate — near latch "
          "only, no on-plate credit, no success, score <= 0.105",
          bool(scene._l_near[0]) and not bool(scene.pot_on_plate()[0])
          and not bool(scene._l_on[0]) and not bool(scene._foul[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.105)

    # ================= 7. swap incomplete: pan left on the plate ================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    side = _pan_side()
    ok_drop = _drop_pot_on_plate(-side * c.pot_slot_x, pot_slot_yaw(side))
    _step(120)
    _report("swap-incomplete")
    check("swap incomplete: pot correctly on the free half but the pan LEFT on "
          "the plate — 0.35 latched, no swap credit, no success",
          ok_drop and bool(scene.pan_on_plate()[0]) and bool(scene._l_on[0])
          and not bool(scene._l_swap[0]) and not bool(scene._foul[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.355
          and float(scene.score()[0]) >= 0.345)

    # ================= 8. tipped pot: upright/z clauses refuse the latch ========================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    side = _pan_side()
    s2 = math.sqrt(0.5)
    q_side = torch.tensor([s2, 0.0, s2, 0.0], device=device).expand(n, 4)  # Ry(90)
    _write_body(scene.pot, _plate_hover(-side * c.pot_slot_x, 0.0, dz=0.040), q_side)
    _step(420)
    _report("tipped-pot")
    check("tipped pot: pot lying on its SIDE on the free half — upright/z clauses "
          "refuse the on-plate latch, no success, score <= 0.105",
          not bool(scene.pot_upright()[0]) and not bool(scene._l_on[0])
          and bool(scene._l_near[0]) and not bool(scene._foul[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.105)

    # ================= 9. pan too close: clearance clause blocks the swap latch =================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    side = _pan_side()
    ok_drop = _drop_pot_on_plate(-side * c.pot_slot_x, pot_slot_yaw(side))
    pan_quat = scene.pan.data.root_quat_w.clone()
    _write_body(scene.pan, _deck_spot(c.housing_pos[0], -0.12), pan_quat)  # 0.22m < 0.25m
    _step(300)
    _report("pan-too-close")
    _REC["on"] = False
    check("pan too close: legal order but the pan set down inside the clearance "
          "radius — not clear, no swap latch, no success, score <= 0.355",
          ok_drop and not bool(scene.pan_on_plate()[0]) and not bool(scene.pan_clear()[0])
          and bool(scene.plate_pressed()[0]) and not bool(scene._foul[0])
          and not bool(scene._l_swap[0]) and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.355)

    # ================= save + verdict ===========================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.flame_keeper")
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
    except BaseException:  # noqa: BLE001 - die fast; Kit teardown would hang until the watchdog
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

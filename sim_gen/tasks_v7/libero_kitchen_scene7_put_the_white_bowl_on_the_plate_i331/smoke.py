"""Smoke / rubric-REJECTION battery for BalanceServeScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts counterweight-then-serve and the
latched credit is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes — above all the SEED task's own end state (bowl set on the
plate, nothing else done) — and that the physical claims the task rests on (the deck
stop, the torque arithmetic, the dummy's insufficiency, the cube-prop cheat and the
clause that closes it, the LIVE judging) are load-bearing. Probes are CONSTRUCTED
settled states (teleport, real physics steps, judge); only check 10 deliberately
builds the success state — to prove that success is judged live and DROPS when the
counterweight is removed.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; empty board rests LEVEL
                           (spring calibration); every present cube inside the bowl;
                           score 0, no success;
   2. randomization      — seeded resets: READBACK bowl xy, weight xy differ
                           (3-seed max-pairwise) and the weight/dummy SIDE swaps;
   3. cube-count subset  — over 10 resets the present-cube count takes >= 2 distinct
                           values in {0..3} (count what you see);
   4. null-policy        — 240 idle steps: score ~0, no success;
   5. SEED STRATEGY      — the seed's whole plan: the loaded bowl set on the plate,
                           nothing else -> the un-counterweighted board TIPS ONTO THE
                           DECK STOP (|tilt| > 3.5 deg, ~5.7), seat credit latches
                           (0.55 cap) but NO success;
   6. wrong radius       — weight released 50 mm off the balance radius: residual
                           torque 0.245 N*m -> board grounds out of the level band;
                           both latches set, score caps at 0.55, no success;
   7. dummy decoy        — the pale 40 g block at its BEST radius instead of the
                           weight: max torque 0.075 N*m vs >= 0.26 N*m load -> the
                           board stays >= ~5 deg off, no success;
   8. missing cube       — one present cube left on the ground, weight tuned to the
                           REDUCED load: the board IS level (non-vacuous!) but the
                           cube-accounting clause refuses -> no success;
   9. CUBE-PROP cheat    — a present cube stood on the deck under the plate half
                           props the tipping board INSIDE the level band (the cheat
                           physically works: |tilt| <= 3.5 deg) — and success still
                           refuses because that cube is neither in the bowl nor in
                           the rack;
  10. LIVE judging       — construct the full success state (assert success True),
                           then teleport the counterweight away: within ~2.5 s the
                           board falls off level and success DROPS; the latched 0.55
                           partial credit remains;
  11. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene7_put_the_white_bowl_on_the_plate_i331.smoke --headless
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

_qz = task_scene._qz

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
    pres = scene.present[0]
    inb = scene.cubes_in_bowl()[0]
    ind = scene.cubes_in_depot()[0]
    marks = "".join(("P" if bool(pres[i]) else "-") + ("B" if bool(inb[i]) else "")
                    + ("D" if bool(ind[i]) else "") + " " for i in range(3))
    tilt = math.degrees(float(scene.tilt()[0]))
    print(f"[smoke] {tag:18s} | tilt={tilt:+.2f}deg cubes {marks}| "
          f"seated={bool(scene.bowl_seated()[0])} ok={bool(scene.cubes_ok()[0])} "
          f"level={bool(scene.level()[0])} place={bool(scene._place[0])} "
          f"seat={bool(scene._seat[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    from isaaclab.utils.math import quat_apply

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.balance_serve")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.15, -0.85, 0.70)) + o),
                                tuple(np.array((0.38, 0.00, 0.10)) + o),
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

    def tilt_deg() -> float:
        _refresh()
        return math.degrees(float(scene.tilt()[0]))

    def beam_pt_w(x: float, y: float, z: float) -> torch.Tensor:
        """LIVE world position of a beam-frame point."""
        _refresh()
        loc = torch.tensor([x, y, z], device=device).expand(n, 3)
        return scene.beam.data.root_pos_w + quat_apply(scene.beam.data.root_quat_w, loc)

    def place_weight(r: float, body=None) -> None:
        """CONSTRUCT `body` (default: the steel weight) released just above the
        weighing half at radius `r` (tracking the live beam pose)."""
        body = body or scene.weight
        h = c.weight_size[2] if body is scene.weight else c.dummy_size[2]
        p = beam_pt_w(-r, 0.0, c.plank_t / 2 + 0.004 + h / 2)
        _write_body(body, p, None)
        _step(300)

    def serve_bowl(cube_idx: list[int]) -> None:
        """CONSTRUCT the bowl released just above the LIVE plate pose with cubes
        `cube_idx` inside it, teleported to the stable 2+1 seats (two abreast on
        the floor + one bridging the seam — same plan as reset)."""
        bp = beam_pt_w(c.plate_x, 0.0, c.plank_t / 2 + c.plate_t + 0.004)
        _write_body(scene.bowl, bp, None)
        seats = [(-0.017, 0.0, 0.002), (0.017, 0.0, 0.002),
                 (0.0, 0.0, c.cube_size + 0.004)]
        for j, i in enumerate(cube_idx):
            sx, sy, sz = seats[j]
            p = bp.clone()
            p[:, 0] += sx
            p[:, 1] += sy
            p[:, 2] += c.bowl_floor_t + c.cube_size / 2 + sz
            _write_body(scene.cubes[scene.CUBE_NAMES[i]], p, None)
        _step(400)

    def present_idx() -> list[int]:
        return [i for i in range(3) if bool(scene.present[0, i])]

    def reset_with_k(base_seed: int, k_min: int) -> int:
        """Reset on seeds base_seed, base_seed+1, ... until >= k_min cubes present."""
        for s in range(base_seed, base_seed + 40):
            torch.manual_seed(s)
            env.reset()
            if int(scene.present[0].sum()) >= k_min:
                return s
        raise AssertionError(f"no seed in [{base_seed}, {base_seed + 40}) has "
                             f">= {k_min} cubes present")

    def r_bal_for(k: int) -> float:
        return c.plate_x * (c.bowl_mass + c.cube_mass * k) / c.weight_mass

    # ================= 1. settle / no-NaN / empty board level =====================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(150)
    _report("settle")
    _REC["on"] = False
    pos, _v = scene._cube_tensors()
    pres = scene.present[0]
    t0 = tilt_deg()
    check("settle/no-NaN: layout settles finite; EMPTY board rests LEVEL "
          "(|tilt| < 1.5 deg — spring calibration); every present cube inside the "
          "bowl, accounting holds; score 0, no success",
          bool(torch.isfinite(pos).all()) and abs(t0) < 1.5
          and bool((scene.cubes_in_bowl()[0] | ~pres).all())
          and bool(scene.cubes_ok()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.bowl.data.root_pos_w[0, :2].clone(),
                scene.weight.data.root_pos_w[0, :2].clone(),
                int(scene.weight_side[0]))

    obs = []
    sides = set()
    for s in range(6):  # 3-seed max-pairwise (GPU RNG deltas can collide pairwise)
        torch.manual_seed(101 + s)
        env.reset()
        _step(10)
        b, w, sd = readback()
        obs.append((b, w))
        sides.add(sd)
        if len(obs) >= 3 and len(sides) == 2:
            break
    d_b = max(float((a[0] - b[0]).norm()) for a in obs for b in obs)
    d_w = max(float((a[1] - b[1]).norm()) for a in obs for b in obs)
    print(f"[smoke] randomization max-pairwise deltas: bowl_xy={d_b * 1000:.1f}mm "
          f"weight_xy={d_w * 1000:.1f}mm sides_seen={sorted(sides)}", flush=True)
    check("randomization-is-real: bowl xy and weight xy readback differ across seeds "
          "and the weight/dummy SIDE swaps",
          d_b > 0.003 and d_w > 0.003 and sides == {-1, 1})

    # ================= 3. cube-count subset =======================================================
    counts = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        counts.add(int(scene.present[0].sum()))
    print(f"[smoke] over 10 resets: present-cube counts {sorted(counts)}", flush=True)
    check("cube-count subset: present count takes >= 2 distinct values in {0..3} "
          "over 10 resets",
          len(counts) >= 2 and counts <= {0, 1, 2, 3})

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not bool(scene.success()[0]))

    # ================= 5. negative: the SEED'S OWN STRATEGY =======================================
    # "Put the white bowl on the plate" — CONSTRUCT the seed's end state: the loaded
    # bowl set on the plate, nothing else done. The un-counterweighted board tips
    # until it grounds on the DECK STOP, far outside the level band. Seat credit
    # latches (partial credit is honest) but success refuses.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    serve_bowl(present_idx())
    _report("seed-strategy")
    t5 = tilt_deg()
    print(f"[smoke] seed strategy: board grounded at {t5:+.2f} deg "
          f"(band +/-{c.level_tol_deg:.1f}, deck stop ~{c.deck_stop_deg:.1f})",
          flush=True)
    check("negative (SEED strategy): loaded bowl set on the plate, nothing else — "
          "board tips onto the deck stop OUT of the level band; seat latch fires, "
          "score caps at 0.55, NO success",
          t5 < -c.level_tol_deg and bool(scene._seat[0])
          and bool(scene.bowl_seated()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.551)
    _REC["on"] = False

    # ================= 6. negative: wrong radius ==================================================
    # Weight released 50 mm off the balance radius: residual torque 0.5*g*0.05 =
    # 0.245 N*m -> equilibrium 6.4 deg -> grounds on the deck stop. Both latches
    # set — and success still refuses (the board is not level).
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    k = len(present_idx())
    r_ok = r_bal_for(k)
    r_wrong = r_ok + 0.05 if r_ok < 0.10 else r_ok - 0.05
    place_weight(r_wrong)
    serve_bowl(present_idx())
    _report("wrong-radius")
    t6 = tilt_deg()
    print(f"[smoke] wrong radius: k={k} r_bal={r_ok * 1000:.0f}mm placed at "
          f"{r_wrong * 1000:.0f}mm -> settled {t6:+.2f} deg", flush=True)
    check("negative (wrong radius): weight 50 mm off the balance radius — board "
          "grounds out of the level band; place+seat latch (0.55 cap), no success",
          abs(t6) > c.level_tol_deg and bool(scene._place[0]) and bool(scene._seat[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.551)
    _REC["on"] = False

    # ================= 7. negative: the dummy decoy ===============================================
    # The pale 40 g block at its BEST radius (0.19 m): max torque 0.075 N*m against
    # a >= 0.26 N*m load torque -> residual >= 0.185 N*m -> >= ~4.8 deg. The decoy
    # is rejected by arithmetic, not by fiat.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    place_weight(0.19, body=scene.dummy)
    serve_bowl(present_idx())
    _report("dummy-decoy")
    t7 = tilt_deg()
    print(f"[smoke] dummy at 190mm: settled {t7:+.2f} deg "
          f"(dummy max torque {c.dummy_mass * 9.81 * 0.19:.3f} N*m vs load "
          f"{(c.bowl_mass + c.cube_mass * len(present_idx())) * 9.81 * c.plate_x:.3f} "
          f"N*m)", flush=True)
    check("negative (dummy decoy): the 40 g dummy at its best radius cannot level "
          "the loaded board (torque-insufficient) — no success",
          t7 < -c.level_tol_deg and not bool(scene.success()[0])
          and float(scene.score()[0]) <= 0.551)
    _REC["on"] = False

    # ================= 8. negative: missing cube (accounting, non-vacuous) ========================
    # One present cube left behind on the ground; the weight tuned to the REDUCED
    # load, so the board settles LEVEL — every geometric clause passes EXCEPT the
    # cube accounting. Proves the count is load-bearing, not decorative.
    s8 = reset_with_k(400, 1)
    _step(30)
    _REC["on"] = True
    pidx = present_idx()
    stray = pidx[-1]
    kept = pidx[:-1]
    place_weight(r_bal_for(len(kept)))
    serve_bowl(kept)
    gp = torch.zeros(n, 3, device=device)
    gp[:, 0] = 0.10
    gp[:, 1] = -0.30
    gp[:, 2] = c.cube_size / 2 + 0.002
    gp += env.iscene.env_origins
    _write_body(scene.cubes[scene.CUBE_NAMES[stray]], gp, None)
    _step(200)
    _report("missing-cube")
    t8 = tilt_deg()
    print(f"[smoke] missing cube (seed {s8}): k={len(pidx)} served {len(kept)} "
          f"-> board {t8:+.2f} deg (LEVEL: the check is non-vacuous)", flush=True)
    check("negative (missing cube): weight tuned to the reduced load — board IS "
          "level, bowl seated, but one present cube sits on the ground: the "
          "accounting clause refuses success",
          abs(t8) <= c.level_tol_deg and bool(scene.bowl_seated()[0])
          and not bool(scene.cubes_ok()[0]) and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 9. the CUBE-PROP cheat is real — and closed ================================
    # A 30 mm cube standing on the deck pokes 8 mm into the plank's sweep: stand a
    # PRESENT cube under the plate half and serve the bowl with no counterweight.
    # The board comes to rest ON the cube INSIDE the level band — the cheat
    # physically works — and success still refuses, because that cube is neither in
    # the bowl nor in the rack.
    s9 = reset_with_k(500, 1)
    _step(30)
    _REC["on"] = True
    pidx = present_idx()
    prop = pidx[-1]
    kept = pidx[:-1]
    pp = torch.zeros(n, 3, device=device)
    pp[:, 0] = c.stand_pos[0] + 0.20   # on the deck, under the plate half
    pp[:, 1] = 0.0
    pp[:, 2] = c.deck_top + c.cube_size / 2 + 0.002
    pp += env.iscene.env_origins
    _write_body(scene.cubes[scene.CUBE_NAMES[prop]], pp, None)
    _step(60)
    serve_bowl(kept)
    _report("cube-prop")
    t9 = tilt_deg()
    print(f"[smoke] cube-prop (seed {s9}): board rests ON the deck cube at "
          f"{t9:+.2f} deg — inside the band; accounting closes the cheat", flush=True)
    check("CUBE-PROP cheat: a present cube stood on the deck props the loaded board "
          "INSIDE the level band (the cheat works physically) — success still "
          "refuses via cube accounting",
          abs(t9) <= c.level_tol_deg and bool(scene.bowl_seated()[0])
          and bool(scene.level()[0]) and not bool(scene.cubes_ok()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.551)
    _REC["on"] = False

    # ================= 10. success is judged LIVE =================================================
    # Construct the full success state (counterweight at the balance radius, loaded
    # bowl seated) — assert success True — then teleport the weight away: the board
    # falls onto the deck stop and success DROPS. Latched partial credit (0.55)
    # remains: history is kept, the goal is not.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    k = len(present_idx())
    place_weight(r_bal_for(k))
    serve_bowl(present_idx())
    _step(300)
    _report("constructed-success")
    t10 = tilt_deg()
    got_success = bool(scene.success()[0]) and float(scene.score()[0]) >= 0.99
    far = torch.zeros(n, 3, device=device)
    far[:, 0] = -0.60
    far[:, 1] = -0.60
    far[:, 2] = c.weight_size[2] / 2 + 0.002
    far += env.iscene.env_origins
    _write_body(scene.weight, far, None)
    _step(300)  # ~2.5 s: the board tips back onto the deck stop
    _report("weight-removed")
    t10b = tilt_deg()
    print(f"[smoke] live judging: success at {t10:+.2f} deg -> weight removed -> "
          f"{t10b:+.2f} deg", flush=True)
    check("LIVE judging: constructed success (weight at the balance radius) is "
          "accepted — and removing the counterweight DROPS success within 2.5 s "
          "(board back on the deck stop); latched 0.55 remains",
          got_success and abs(t10) <= c.level_tol_deg
          and t10b < -c.level_tol_deg and not bool(scene.success()[0])
          and 0.549 <= float(scene.score()[0]) <= 0.551)
    _REC["on"] = False

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.balance_serve")
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
    try:
        main()
    except BaseException:  # noqa: BLE001 — Kit teardown hangs; die loudly instead
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(1)

"""Smoke / rubric-REJECTION battery for ValveBeamStoveScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts the correct outcome on
seeds 0 and 1 and the score is monotone along a real trajectory). This battery
proves the rubric REJECTS wrong outcomes, and that the claims the task rests
on — the counterweight statics, the load-sensing success clause (pressing the
beam is not success), the mass discrimination between steel and wood, and the
currently-in-basket credit — are load-bearing. Every probe is CONSTRUCTED
(teleport, real physics steps, judge) — instrumentation, never a solution: no
probe here reaches success() (audited at every step; no probe ever assembles
both steels in the basket).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; beam rests on the CLOSED
                          stop, basket empty, all 4 cubes on the apron row;
                          score 0, no success;
   2. randomization     — two seeded resets: READBACK steel0 xy+yaw and wood0 xy
                          all differ;
   3. permutation       — over 8 resets steel0 occupies >= 3 DISTINCT slots
                          (the 4-slot shuffle is real);
   4. null-policy       — 240 idle steps: score ~0, no success, beam stays on the
                          closed stop (no creep);
   5. SEED-NAIVE press  — the seed's move, "actuate the fixture directly": an
                          external torque provably drives the beam past theta_on
                          (readback — non-vacuous) with the basket EMPTY; the
                          load-sensing clause refuses success THROUGHOUT, score
                          stays 0; released, gravity swings it back CLOSED;
   6. DECOY load        — one steel + both wood blocks constructed IN the basket
                          (settled): the beam stays CLOSED (mass discrimination
                          from below) -> count 1, score 0.25, no success; then
                          pressing the under-loaded beam past theta_on is STILL
                          refused, and it re-closes on release;
   7. wrong place       — both steels ON the stove (one on the burner plate, one
                          on the counterweight arm): touching the fixture/beam is
                          not loading the basket -> count 0, beam closed, score 0;
   8. latch honesty     — one steel seated (count 1, 0.25), then lifted back OUT
                          to the apron: credit requires CURRENTLY in-basket ->
                          count 0, score ~0 (the latch alone can never score);
   9. rejection audit   — success() was never True at ANY step of the battery;
  10. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_i280.smoke --headless
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
import traceback  # noqa: E402

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
_WDT = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                        os._exit(3)))
_WDT.daemon = True
_WDT.start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"hit": False}


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate body vectors `v` (N,3) into world by quats `q` (N,4 wxyz)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["hit"] = _AUDIT["hit"] or bool(env.scene.success().any())
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


def _write_state(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None) -> None:
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
    ins = scene.steels_in_basket()[0]
    print(f"[smoke] {tag:18s} | count={int(scene.count()[0])} "
          f"in={[int(b) for b in ins]} "
          f"beam={math.degrees(float(scene.beam_angle()[0])):+.1f}deg "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.valve_beam_stove")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.80, -0.95, 0.85)) + o),
                                tuple(np.array((-0.02, 0.05, 0.25)) + o),
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

    def beam_deg() -> float:
        _refresh()
        return math.degrees(float(scene.beam_angle()[0]))

    def stove_w() -> torch.Tensor:
        _refresh()
        return scene.stove.data.root_pos_w[0]

    def put_rel(body, x: float, y: float, z: float, quat=None) -> None:
        """CONSTRUCT: write the body at a stove-local point (probe
        instrumentation; the caller settles when the arrangement is done)."""
        t = stove_w()
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0] = float(t[0]) + x
        pos[:, 1] = float(t[1]) + y
        pos[:, 2] = float(t[2]) + z
        _write_state(body, pos, quat)

    def put_in_basket(body, off_y: float, off_z: float = 0.008) -> None:
        """CONSTRUCT: set a cube just above the basket floor at the beam's
        CURRENT tilt (beam-frame placement, beam-aligned orientation)."""
        _refresh()
        q = scene.beam.data.root_quat_w
        off = torch.tensor([[c.basket_x, off_y, off_z]], device=device).expand(n, 3)
        pos = scene.beam.data.root_pos_w + _quat_rotate(q, off)
        _write_state(body, pos, q.clone())

    def zero_beam_wrench() -> None:
        scene.beam.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=device), torch.zeros(n, 1, 3, device=device))

    def press_beam(tau: float, hold: int = 120) -> tuple[float, bool]:
        """Drive the beam open with an external hinge torque (the beam only
        rotates about y, so the body-frame y torque IS the hinge torque).
        Returns (max angle reached in deg, success ever True while driven).
        The caller must zero + settle afterwards."""
        max_deg, hit = -90.0, False
        f0 = torch.zeros(n, 1, 3, device=device)
        tq = torch.zeros(n, 1, 3, device=device)
        tq[:, 0, 1] = tau
        held = 0
        for _ in range(500):
            scene.beam.set_external_force_and_torque(f0, tq)
            _step(1)
            d = beam_deg()
            max_deg = max(max_deg, d)
            hit = hit or bool(scene.success()[0])
            if d >= c.theta_on_deg:
                held += 1
                if held >= hold:
                    break
        return max_deg, hit

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    _refresh()
    fin = bool(torch.isfinite(scene.beam.data.root_pos_w).all())
    on_row = True
    for b in [*scene.steels, *scene.woods]:
        fin = fin and bool(torch.isfinite(b.data.root_pos_w).all())
        py = float((b.data.root_pos_w - scene.env_origins)[0, 1])
        on_row = on_row and abs(py - c.slot_y) < c.slot_jitter + 0.02
    check("settle/no-NaN: beam rests on the CLOSED stop, basket empty, all 4 cubes "
          "on the apron row; score 0, no success",
          fin and on_row and beam_deg() <= c.beam_lo_deg + 3.0
          and int(scene.count()[0]) == 0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.steels[0].data.root_pos_w[0, :2].clone(),
                yaw_of(scene.steels[0].data.root_quat_w[0]),
                scene.woods[0].data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_s, a_y, a_w = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_s, b_y, b_w = readback()
    d_s, d_y, d_w = float((a_s - b_s).norm()), dyaw(a_y, b_y), float((a_w - b_w).norm())
    print(f"[smoke] randomization deltas: steel0_xy={d_s * 1000:.1f}mm "
          f"steel0_yaw={d_y:.1f}deg wood0_xy={d_w * 1000:.1f}mm", flush=True)
    check("randomization-is-real: steel0 xy, steel0 yaw, wood0 xy readback differ",
          d_s > 0.005 and d_y > 3.0 and d_w > 0.005)

    # ================= 3. permutation: steel0 visits >= 3 distinct slots ==========================
    slots_seen = set()
    xs = torch.tensor(c.slot_xs)
    for s in range(8):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        sx = float((scene.steels[0].data.root_pos_w - scene.env_origins)[0, 0])
        slot = int(torch.argmin((xs - sx).abs()))
        assert abs(float(xs[slot]) - sx) < c.slot_jitter + 0.02, \
            "steel0 must land on one of the 4 slots"
        slots_seen.add(slot)
    print(f"[smoke] steel0 slots over 8 resets: {sorted(slots_seen)}", flush=True)
    check("permutation: steel0 occupies >= 3 distinct slots over 8 resets "
          "(the 4-slot shuffle is real)", len(slots_seen) >= 3)

    # ================= 4. null policy fails (and the beam does not creep) =========================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, beam stays on "
          "the closed stop",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0])
          and int(scene.count()[0]) == 0 and beam_deg() <= c.beam_lo_deg + 3.0)

    # ================= 5. SEED-NAIVE: actuate the fixture directly ================================
    # The seed's whole plan is "grab the fixture and actuate it past the angle
    # threshold". Here that move is REFUSED: an external torque drives the beam
    # past theta_on with the basket EMPTY (the actuator provably moved —
    # readback), yet the load-sensing clause keeps success False and score 0
    # the whole time; released, the counterweight swings it back CLOSED.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    max_deg, hit = press_beam(2.0)
    pressed_success = bool(scene.success()[0])
    pressed_score = float(scene.score()[0])
    zero_beam_wrench()
    _step(360)  # hands-off: gravity must re-close the valve
    _report("seed-naive-press")
    _REC["on"] = False
    print(f"[smoke] press: max={max_deg:+.1f}deg pressed_success={pressed_success} "
          f"pressed_score={pressed_score:.3f} released={beam_deg():+.1f}deg", flush=True)
    check("SEED-NAIVE press: the beam provably driven past theta_on with an empty "
          "basket — success False and score 0 THROUGHOUT; released, gravity "
          "re-closes it",
          max_deg >= c.theta_on_deg and not hit and not pressed_success
          and pressed_score <= 0.01
          and beam_deg() <= c.beam_lo_deg + 4.0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 6. DECOY load + under-loaded press =========================================
    # One steel and BOTH wood blocks in the basket: the counterweight must win
    # (cfg-asserted >= 15 % margin) — the beam stays closed. Then pressing the
    # under-loaded beam past theta_on is still refused (needs BOTH steels), and
    # it re-closes on release.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    put_in_basket(scene.steels[0], -0.022)
    _step(90)
    put_in_basket(scene.woods[0], 0.022)
    _step(90)
    put_in_basket(scene.woods[1], 0.022, off_z=c.cube_s + 0.012)  # stacked on wood0
    _step(240)
    _report("decoy-load")
    closed_under_decoys = beam_deg() <= c.beam_lo_deg + 4.0
    decoy_count = int(scene.count()[0])
    decoy_score = float(scene.score()[0])
    max_deg2, hit2 = press_beam(2.0)
    pressed2 = bool(scene.success()[0])
    zero_beam_wrench()
    _step(360)
    _report("decoy-press")
    _REC["on"] = False
    print(f"[smoke] decoy: closed={closed_under_decoys} count={decoy_count} "
          f"press_max={max_deg2:+.1f}deg released={beam_deg():+.1f}deg", flush=True)
    check("DECOY load: one steel + both woods in the basket cannot sink the beam "
          "(count 1, score 0.25, no success); pressing the under-loaded beam past "
          "theta_on is STILL refused and it re-closes on release",
          closed_under_decoys and decoy_count == 1
          and abs(decoy_score - 0.25) < 0.01
          and max_deg2 >= c.theta_on_deg and not hit2 and not pressed2
          and beam_deg() <= c.beam_lo_deg + 4.0
          and not bool(scene.success()[0]))

    # ================= 7. wrong place: steels touching stove/beam but not in the basket ===========
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    # one steel on the burner plate (the "put metal on the stove" naive move) ...
    put_rel(scene.steels[0], c.burner_x, 0.0, c.burner_h + 0.004 + c.cube_s / 2)
    # ... and one on the counterweight arm (touching the BEAM is not loading it)
    _refresh()
    qb = scene.beam.data.root_quat_w
    off = torch.tensor([[-c.arm_in + 0.035, 0.0, 0.025 + c.cube_s / 2 + 0.004]],
                       device=device).expand(n, 3)
    _write_state(scene.steels[1], scene.beam.data.root_pos_w + _quat_rotate(qb, off),
                 qb.clone())
    _step(300)
    _report("wrong-place")
    _REC["on"] = False
    check("wrong place: steels ON the burner plate and ON the counterweight arm are "
          "not in the basket -> count 0, beam closed, score 0, no success",
          int(scene.count()[0]) == 0 and beam_deg() <= c.beam_lo_deg + 4.0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 8. latch honesty: credit requires CURRENTLY in-basket ======================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    put_in_basket(scene.steels[0], -0.022)
    _step(150)
    assert bool(scene.counted()[0, 0]), "the seated steel must latch and count first"
    assert abs(float(scene.score()[0]) - 0.25) < 0.01, "one seated steel = 0.25"
    # lift it back OUT to a free apron spot (the reverse pick-and-place)
    put_rel(scene.steels[0], 0.30, -0.36, 0.004 + c.cube_s / 2, _qz(
        torch.zeros(n, device=device)))
    _step(120)
    _report("latch-honesty")
    _REC["on"] = False
    check("latch honesty: the steel lifted back OUT of the basket loses its credit "
          "-> count 0, score ~0 (the latch alone can never score)",
          int(scene.count()[0]) == 0 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 9. rejection audit =========================================================
    check("rejection audit: success() was never True at ANY step of the battery",
          not _AUDIT["hit"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.valve_beam_stove")
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — fail fast, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)

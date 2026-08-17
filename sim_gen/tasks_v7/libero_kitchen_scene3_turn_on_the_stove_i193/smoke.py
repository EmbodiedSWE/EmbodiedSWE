"""Smoke / rubric-REJECTION battery for StokerStoveScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct outcome and the
count is monotone along a real trajectory). This battery proves the rubric
REJECTS wrong outcomes, and that the claims the task rests on — the grate's
drop-in denial, the one-way flap's escape denial, the inside window, the
butter-exclusion clause and the calm gate — are load-bearing. Every probe is
CONSTRUCTED (teleport, real physics steps, judge) — instrumentation, never a
solution: no probe here reaches success() (audited at every step; no probe
ever assembles 3 clean briquettes inside).

Checks:
   1. settle/no-NaN     — seeded reset settles finite; flap hangs shut, firebox
                          empty, all 5 items on the apron row; score 0, no success;
   2. randomization     — two seeded resets: READBACK pellet0 xy+yaw and butter xy
                          all differ;
   3. permutation       — over 8 resets the butter occupies >= 3 DISTINCT slots
                          (the 5-slot shuffle is real);
   4. null-policy       — 240 idle steps: score ~0, no success, flap stays shut;
   5. SEED-NAIVE        — fuel piled ON TOP of the stove (3 briquettes set on the
                          grate — the "put it on the stove" move): they rest ON the
                          bars, none inside -> score ~0, no success;
   6. grate drop-in     — a briquette DROPPED onto the grate from above: never
                          inside at any substep, comes to rest ON the bars (the
                          16 mm gaps deny it; readback height proves it);
   7. near-miss         — 2 briquettes inside + 1 settled at the port MOUTH (in
                          the chute, 2 cm short) -> count 2, score 0.50, no success;
   8. CONTAMINANT       — butter inside FIRST, then 3 briquettes in: the spoilage
                          clause refuses -> no success, score <= 0.16 (x0.2);
   9. butter-only       — only the butter inside -> score ~0, no success;
  10. ONE-WAY flap      — a briquette inside, force-pushed OUTWARD at port height
                          (gravity cancelled): it provably reaches and loads the
                          flap, the hinge stop denies outward swing, it never
                          exits; released, it is still counted -> count 1, 0.25;
  11. MECHANISM         — one briquette fed the solve's way (servo push along the
                          chute): the flap visibly swings > 15 deg, the briquette
                          drops in by contact -> count 1, score 0.25, no success;
  12. rejection audit   — success() was never True at ANY step of the battery;
  13. frames.npz        — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.libero_kitchen_scene3_turn_on_the_stove_i193.smoke --headless
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
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_AUDIT = {"hit": False}


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
    ins = scene.pellets_inside()[0]
    print(f"[smoke] {tag:18s} | count={int(scene.count()[0])} "
          f"inside={[int(b) for b in ins]} "
          f"butter_in={bool(scene.butter_inside()[0])} "
          f"flap={math.degrees(float(scene.flap_angle()[0])):+.1f}deg "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.stoker_stove")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((0.75, -1.00, 0.80)) + o),
                                tuple(np.array((-0.08, 0.00, 0.22)) + o),
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

    def stove_w() -> torch.Tensor:
        _refresh()
        return scene.stove.data.root_pos_w[0]

    def rel(body) -> torch.Tensor:
        _refresh()
        return (body.data.root_pos_w - scene.stove.data.root_pos_w)[0]

    def flap_deg() -> float:
        _refresh()
        return math.degrees(float(scene.flap_angle()[0]))

    def put_rel(body, x: float, y: float, z: float, quat=None) -> None:
        """CONSTRUCT: write the body at a stove-local point (probe
        instrumentation; the caller settles when the arrangement is done)."""
        t = stove_w()
        pos = torch.zeros(n, 3, device=device)
        pos[:, 0] = float(t[0]) + x
        pos[:, 1] = float(t[1]) + y
        pos[:, 2] = float(t[2]) + z
        _write_state(body, pos, quat)

    # spots used to CONSTRUCT briquettes/butter inside the firebox (stove-local;
    # resting on the interior floor, clear of walls, flap sweep and each other)
    pel_z = c.floor_t + c.pellet_s / 2 + 0.004
    but_z = c.floor_t + c.butter_lz / 2 + 0.004
    spots = [(-0.05, 0.03), (0.0, -0.03), (0.05, 0.03)]
    butter_spot = (0.0, 0.045)

    def construct_inside(k: int, spot: tuple) -> None:
        put_rel(scene.pellets[k], spot[0], spot[1], pel_z)

    # the solve's feed move (servo push along the chute) — used by MECHANISM
    def feed_one(k: int) -> float:
        p = scene.pellets[k]
        t = stove_w()
        start = torch.zeros(n, 3, device=device)
        start[:, 0] = t[0]
        start[:, 1] = t[1] - 0.190
        start[:, 2] = t[2] + c.sill_z + c.pellet_s / 2 + 0.002
        _write_state(p, start)
        _step(20)
        max_flap = 0.0
        for _ in range(700):
            vel = float(p.data.root_lin_vel_w[0, 1])
            r = rel(p)
            if float(r[1]) > -0.055 or bool(scene.pellets_inside()[0, k]):
                break
            f = torch.zeros(n, 1, 3, device=device)
            f[:, 0, 0] = max(-0.15, min(0.15, 4.0 * (float(t[0]) - float(p.data.root_pos_w[0, 0]))))
            f[:, 0, 1] = max(-1.0, min(1.0, 2.5 * (0.30 - vel)))
            tau = -4.0e-4 * p.data.root_ang_vel_w.view(n, 1, 3)
            p.set_external_force_and_torque(f, tau)
            _step(1)
            max_flap = max(max_flap, flap_deg())
        p.set_external_force_and_torque(
            torch.zeros(n, 1, 3, device=device), torch.zeros(n, 1, 3, device=device))
        _step(90)
        return max_flap

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(60)
    _report("settle")
    _REC["on"] = False
    _refresh()
    fin = bool(torch.isfinite(scene.flap.data.root_pos_w).all()
               and torch.isfinite(scene.butter.data.root_pos_w).all())
    on_row = True
    for p in [*scene.pellets, scene.butter]:
        fin = fin and bool(torch.isfinite(p.data.root_pos_w).all())
        py = float((p.data.root_pos_w - scene.env_origins)[0, 1])
        on_row = on_row and abs(py - c.slot_y) < c.slot_jitter + 0.02
    check("settle/no-NaN: flap hangs shut, firebox empty, all 5 items on the apron "
          "row; score 0, no success",
          fin and on_row and abs(flap_deg()) < 3.0 and int(scene.count()[0]) == 0
          and not bool(scene.butter_inside()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (scene.pellets[0].data.root_pos_w[0, :2].clone(),
                yaw_of(scene.pellets[0].data.root_quat_w[0]),
                scene.butter.data.root_pos_w[0, :2].clone())

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_p, a_y, a_b = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_p, b_y, b_b = readback()
    d_p, d_y, d_b = float((a_p - b_p).norm()), dyaw(a_y, b_y), float((a_b - b_b).norm())
    print(f"[smoke] randomization deltas: pellet0_xy={d_p * 1000:.1f}mm "
          f"pellet0_yaw={d_y:.1f}deg butter_xy={d_b * 1000:.1f}mm", flush=True)
    check("randomization-is-real: pellet0 xy, pellet0 yaw, butter xy readback differ",
          d_p > 0.005 and d_y > 3.0 and d_b > 0.005)

    # ================= 3. permutation: butter visits >= 3 distinct slots ==========================
    slots_seen = set()
    xs = torch.tensor(c.slot_xs)
    for s in range(8):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        bx = float((scene.butter.data.root_pos_w - scene.env_origins)[0, 0])
        slot = int(torch.argmin((xs - bx).abs()))
        assert abs(float(xs[slot]) - bx) < c.slot_jitter + 0.02, \
            "butter must land on one of the 5 slots"
        slots_seen.add(slot)
    print(f"[smoke] butter slots over 8 resets: {sorted(slots_seen)}", flush=True)
    check("permutation: the butter occupies >= 3 distinct slots over 8 resets "
          "(the 5-slot shuffle is real)", len(slots_seen) >= 3)

    # ================= 4. null policy fails (and the flap does not creep) =========================
    torch.manual_seed(100)
    env.reset()
    _step(240)
    _report("null-policy")
    check("null-policy-fails: 240 idle steps, score ~0, no success, flap stays shut",
          float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0])
          and int(scene.count()[0]) == 0 and abs(flap_deg()) < 3.0)

    # ================= 5. SEED-NAIVE: fuel piled ON the stove =====================================
    # The naive read of "turn on the stove with fuel": set the briquettes ON the
    # stove top. They rest on the grate — never inside.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    for k, x in enumerate((-0.06, 0.0, 0.06)):
        put_rel(scene.pellets[k], x, 0.0, c.fire_h + c.pellet_s / 2 + 0.004)
    _step(150)
    _report("seed-naive")
    _REC["on"] = False
    on_top = all(float(rel(scene.pellets[k])[2]) > 0.150 for k in range(3))
    check("SEED-NAIVE: 3 briquettes set ON the stove top rest on the grate bars — "
          "none inside: score ~0, no success",
          on_top and int(scene.count()[0]) == 0
          and not bool(scene.pellets_inside()[0].any())
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 6. grate drop-in denial ====================================================
    # A briquette dropped from above the grate centre: audited every substep —
    # it must NEVER be inside, and it comes to rest ON the bars.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    put_rel(scene.pellets[0], 0.01, 0.01, c.fire_h + 0.060)
    ever_inside = False
    for _ in range(180):
        _step(1)
        ever_inside = ever_inside or bool(scene.pellets_inside()[0, 0])
    _report("grate-drop")
    _REC["on"] = False
    rz = float(rel(scene.pellets[0])[2])
    print(f"[smoke] grate-drop: rest z_rel={rz:.3f} (grate top "
          f"{c.grate_zc + c.grate_bar_t / 2:.3f}) ever_inside={ever_inside}", flush=True)
    check("grate drop-in denial: a briquette dropped onto the grate is NEVER inside "
          "at any substep and rests ON the bars",
          not ever_inside and rz > 0.150 and int(scene.count()[0]) == 0
          and float(scene.score()[0]) <= 0.01)

    # ================= 7. near-miss: 2 in + 1 at the port mouth ===================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    construct_inside(0, spots[0])
    construct_inside(1, spots[1])
    _step(120)  # settle + latch (inside AND calm)
    # third briquette parked at the port MOUTH: on the chute floor, 2 cm short
    put_rel(scene.pellets[2], 0.0, -0.105, c.sill_z + c.pellet_s / 2 + 0.004)
    _step(120)
    _report("near-miss")
    _REC["on"] = False
    r2y = float(rel(scene.pellets[2])[1])
    check("near-miss: 2 briquettes inside + 1 settled at the port mouth -> count 2, "
          "score 0.50, mouth briquette NOT counted, no success",
          int(scene.count()[0]) == 2 and not bool(scene.pellets_inside()[0, 2])
          and r2y < -0.085
          and abs(float(scene.score()[0]) - 0.50) < 0.01
          and not bool(scene.success()[0]))

    # ================= 8. CONTAMINANT: butter inside spoils the firebox ===========================
    # Butter FIRST (so no prefix of the construction ever satisfies the goal),
    # then 3 briquettes — the quota alone would succeed, the butter must refuse.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    put_rel(scene.butter, butter_spot[0], butter_spot[1], but_z)
    _step(90)
    assert bool(scene.butter_inside()[0]), "constructed butter must read back inside"
    for k, sp in enumerate(spots):
        construct_inside(k, sp)
    _step(150)
    _report("contaminant")
    _REC["on"] = False
    check("CONTAMINANT: 3 briquettes counted inside but the butter is in too — "
          "spoilage refuses: no success, score <= 0.16",
          int(scene.count()[0]) == 3 and bool(scene.butter_inside()[0])
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.16)

    # ================= 9. butter-only =============================================================
    torch.manual_seed(100)
    env.reset()
    _step(30)
    put_rel(scene.butter, butter_spot[0], butter_spot[1], but_z)
    _step(90)
    _report("butter-only")
    check("butter-only: just the butter inside -> score ~0, no success",
          bool(scene.butter_inside()[0]) and int(scene.count()[0]) == 0
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 10. ONE-WAY flap: no way back out ==========================================
    # A briquette inside, then force-pushed OUTWARD at port height (gravity
    # cancelled so the below-sill wall cannot shadow the flap): it must really
    # reach and load the flap (displacement audited — non-vacuous), the hinge
    # stop must deny outward swing, and it must never exit. Released, it is
    # still counted.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    construct_inside(0, (0.0, -0.03))
    _step(120)
    assert bool(scene.counted()[0, 0]), "the constructed briquette must latch first"
    p = scene.pellets[0]
    put_rel(p, 0.0, -0.030, c.sill_z + c.pellet_s / 2 + 0.004)  # port height, inside
    min_y, min_flap, escaped = 1.0, 90.0, False
    fz = c.pellet_mass * 9.81
    for _ in range(300):
        f = torch.zeros(n, 1, 3, device=device)
        f[:, 0, 1] = -0.60          # outward push
        f[:, 0, 2] = fz             # cancel gravity: load the FLAP, not the sill wall
        p.set_external_force_and_torque(f, torch.zeros(n, 1, 3, device=device))
        _step(1)
        r = rel(p)
        min_y = min(min_y, float(r[1]))
        min_flap = min(min_flap, flap_deg())
        escaped = escaped or float(r[1]) < -0.086
    p.set_external_force_and_torque(
        torch.zeros(n, 1, 3, device=device), torch.zeros(n, 1, 3, device=device))
    _step(120)
    _report("one-way")
    _REC["on"] = False
    print(f"[smoke] one-way: min_rel_y={min_y * 1000:.1f}mm (flap plane "
          f"{c.flap_y * 1000:.1f}mm) min_flap={min_flap:.1f}deg escaped={escaped}",
          flush=True)
    check("ONE-WAY flap: the outward-pushed briquette really loads the flap "
          "(travelled to it), the hinge stop holds (no outward swing), it never "
          "exits and is still counted after release",
          (min_y < -0.055) and (min_flap > c.flap_lo_deg - 1.5) and not escaped
          and bool(scene.counted()[0, 0]) and int(scene.count()[0]) == 1
          and abs(float(scene.score()[0]) - 0.25) < 0.01
          and not bool(scene.success()[0]))

    # ================= 11. MECHANISM: feeding really goes through the flap ========================
    # The solve's move from a fresh reset, fully instrumented: servo-push one
    # briquette along the chute. The flap must VISIBLY swing (the actuator
    # provably moved — non-vacuous) and the briquette must end up counted.
    torch.manual_seed(100)
    env.reset()
    _step(30)
    _REC["on"] = True
    max_flap = feed_one(0)
    _report("mechanism")
    _REC["on"] = False
    print(f"[smoke] mechanism: max_flap={max_flap:.1f}deg", flush=True)
    check("MECHANISM: one briquette fed by the servo push — the flap swings "
          "> 15 deg and the briquette is counted: count 1, score 0.25, no success",
          max_flap > 15.0 and bool(scene.counted()[0, 0])
          and int(scene.count()[0]) == 1
          and abs(float(scene.score()[0]) - 0.25) < 0.01
          and not bool(scene.success()[0]))

    # ================= 12. rejection audit ========================================================
    check("rejection audit: success() was never True at ANY step of the battery",
          not _AUDIT["hit"])

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.stoker_stove")
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
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — fail fast, never idle until the watchdog
        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)

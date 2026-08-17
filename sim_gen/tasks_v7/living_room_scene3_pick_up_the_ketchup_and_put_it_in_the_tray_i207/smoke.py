"""Smoke / rubric-REJECTION battery for QueueDispenserScene — NullRobot, teleported probes.

solve.py is the acceptance proof (the rubric accepts a correct reject-stage-serve
episode with monotone latched credit, several seeds). This battery proves the rubric
REJECTS wrong outcomes and that the geometry claims the task rests on — the queue is
captive, the lip meters one carton per push and retains the dropped follower, the pit
is a point of no return, a dirty tray never scores — are physics, not fiat. Every
probe is CONSTRUCTED as a settled state (teleport, real physics steps, judge) —
instrumentation, never a solution. Smoke constructs may write bodies inside rubric
volumes (that is the point of a rubric probe); only solve.py is barred from doing so.

Checks:
   1. settle/no-NaN      — seeded reset settles finite; all three cartons queued in
                           the magazine, tray loose on the ground; score ~0, no success;
   2. randomization      — two seeded resets: READBACK station yaw, station xy and
                           the tray's start pose all differ;
   3. queue shuffle      — over 10 resets the ketchup occupies >= 2 different queue
                           levels (readback of the shuffled permutation);
   4. null-policy        — 240 idle steps: cartons stay queued, tray stays put
                           (< 8 mm), score ~0, no success;
   5. SEED STRATEGY dead — the seed's plan starts with "pick up the ketchup": a 10 N
                           upward yank demonstrably lifts the queue (>= 8 mm, non-
                           vacuous) yet the ROOF keeps the carton in the magazine,
                           and an 8 N rearward pull is stopped by the rear wall (the
                           44 mm slot < the 50 mm carton). The ketchup cannot be
                           acquired, so the seed's carry-and-place plan is dead;
   6. lip holds a nudge  — 1.5 N quasi-static push on the bottom carton for 2.5 s:
                           it stays behind the lip, in the magazine, score ~0
                           (non-vacuous by pairing: check 7 ejects with the SAME
                           force encoding at solve-level force);
   7. one-per-push       — a solve-style push (4 -> 12 N) on the bottom DISTRACTOR
                           with no tray staged: exactly that carton ends in the
                           REJECT PIT; the other two remain queued in the magazine
                           (the dropped follower re-seats behind the lip);
   8. ORDER violation    — ketchup pushed out with NO tray staged: it falls into the
                           pit; staging the tray afterwards cannot help — no
                           success, score stays at latched partial credit;
   9. wrong object       — BUTTER constructed inside the seated tray (ketchup still
                           queued): no success; then the ketchup added on top of the
                           butter: STILL no success (dirty tray) — no prefix of this
                           construction ever scores 1;
  10. unseated serve     — ketchup constructed inside the tray ON THE GROUND: served
                           credit only (~0.35), no success (tray not on the rim);
  11. follower pile      — ketchup in the seated tray AND butter piled ON TOP of it:
                           in_tray(ketchup) and tray_seated both hold, yet success
                           is False — the dirty gate covers the tray's whole column
                           of space (regression for the follower-pileup loophole);
  12. frames.npz         — video captured and saved to the CWD.

Prints exactly `SIM_GEN_SMOKE: ALL PASS <n>/<n>` when every check passes; hard exit.

Run: python -m simgen_tasks.living_room_scene3_pick_up_the_ketchup_and_put_it_in_the_tray_i207.smoke --headless
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
    from .scene import encode_force
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scene import encode_force

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": False, "annot": None, "frames": [], "i": 0}
_MODE = {"m": 0}  # force-frame convention, pinned by the probe in main()


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


def _write_body(body, pos_w: torch.Tensor, quat: torch.Tensor | None = None,
                lin_vel_w: torch.Tensor | None = None) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = quat
    if lin_vel_w is not None:
        st[:, 7:10] = lin_vel_w
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    tl = scene.station_local(scene.tray.data.root_pos_w)[0]
    parts = []
    for name, body in zip(scene.ITEMS, scene.items):
        bl = scene.station_local(body.data.root_pos_w)[0]
        parts.append(f"{name[:4]}=({float(bl[0]):+.3f},{float(bl[1]):+.3f},"
                     f"{float(bl[2]):+.3f})")
    print(f"[smoke] {tag:16s} | tray=({float(tl[0]):+.3f},{float(tl[1]):+.3f},"
          f"{float(tl[2]):+.3f}) " + " ".join(parts) +
          f" seated={bool(scene.tray_seated()[0])} in={bool(scene.in_tray(scene.ketchup)[0])} "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


def _force_on(body, f_world: torch.Tensor, q_ref: torch.Tensor) -> None:
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    f = encode_force(_MODE["m"], q_ref, body.data.root_quat_w,
                     f_world.expand(n, 3)).view(n, 1, 3)
    body.set_external_force_and_torque(f, zero, env_ids=_all_ids(), is_global=True)


def _force_off(body) -> None:
    n = _ENV.num_envs
    zero = torch.zeros(n, 1, 3, device=_ENV.device)
    body.set_external_force_and_torque(zero, zero, env_ids=_all_ids())


def _hold_force(body, f_world: torch.Tensor, steps: int) -> None:
    """Constant world-frame force at the CoM for `steps` steps, then clear."""
    q_ref = body.data.root_quat_w.clone()
    for _ in range(steps):
        _force_on(body, f_world, q_ref)
        _step(1)
    _force_off(body)


def _dispense_push(body, fmax: float = 12.0, max_steps: int = 900) -> bool:
    """solve.py's rear-slot push: bang-bang station-local +x CoM force with stall
    escalation, released once the carton is past the lip (station x > 0.095)."""
    scene = _ENV.scene
    from isaaclab.utils.math import quat_apply

    device = _ENV.device
    fdir = quat_apply(scene._st_quat,
                      torch.tensor([1.0, 0.0, 0.0], device=device).expand(_ENV.num_envs, 3))
    q_ref = body.data.root_quat_w.clone()
    fmag = 4.0
    probe_x = float(scene.station_local(body.data.root_pos_w)[0, 0])
    out = False
    for i in range(max_steps):
        x = float(scene.station_local(body.data.root_pos_w)[0, 0])
        if x > 0.095:
            out = True
            break
        v = float(body.data.root_lin_vel_w[0].norm())
        mag = fmag if v < 0.15 else (0.4 * fmag if v < 0.30 else 0.0)
        if mag > 0.0:
            _force_on(body, mag * fdir[0], q_ref)
        else:
            _force_off(body)
        _step(1)
        if i % 45 == 44:
            x_new = float(scene.station_local(body.data.root_pos_w)[0, 0])
            if x_new < probe_x + 0.002:
                fmag = min(fmag + 1.5, fmax)
            probe_x = x_new
    _force_off(body)
    return out


def _reset_with(pred, base: int, tries: int = 60) -> int:
    """Reset with manual seeds base, base+1, ... until `pred()` holds; returns the
    seed used. Dies loudly if none matches (queue shuffle would have to be broken)."""
    for s in range(tries):
        torch.manual_seed(base + s)
        _ENV.reset()
        _refresh()
        if pred():
            return base + s
    print(f"SIM_GEN_SMOKE: FAIL (no seed in [{base},{base + tries}) matches)", flush=True)
    os._exit(4)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.queue_dispenser")().build(num_envs=args.num_envs,
                                                    device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.55, -0.90, 0.85)) + o),
                                tuple(np.array((0.12, 0.00, 0.28)) + o),
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

    def st_world(loc_xyz) -> torch.Tensor:
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.station_world(loc)

    def tray_world(loc_xyz) -> torch.Tensor:
        _refresh()
        loc = torch.tensor(loc_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.tray.data.root_pos_w + quat_apply(scene.tray.data.root_quat_w, loc)

    def x_dir_w() -> torch.Tensor:
        _refresh()
        return quat_apply(scene._st_quat,
                          torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3))[0]

    def loc_of(body) -> torch.Tensor:
        _refresh()
        return scene.station_local(body.data.root_pos_w)[0]

    def bottom_item() -> int:
        return int(scene.levels[0].argmin())

    def stage_tray() -> None:
        """Transport the tray to free air above the seat; contact seats it."""
        _write_body(scene.tray, st_world((c.seat_x, 0.0, c.seat_z[1] + 0.005)),
                    scene._st_quat)
        _step(180)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(100)
    env.reset()
    _REC["on"] = True
    _step(120)
    _report("settle")
    _REC["on"] = False
    in_mag = all(bool(scene.in_magazine(b)[0]) for b in scene.items)
    check("settle/no-NaN: layout settles finite; all three cartons queued in the "
          "magazine, tray loose on the ground; score ~0, no success",
          bool(scene._finite()[0]) and in_mag and not bool(scene.tray_seated()[0])
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # --- force-frame probe (pins encode_force mode for every later force probe):
    # 5 N station-local +x on the tray on open ground must slide it +x.
    tx0 = float(loc_of(scene.tray)[0])
    _hold_force(scene.tray, 5.0 * x_dir_w(), 60)
    _step(30)
    dx = float(loc_of(scene.tray)[0]) - tx0
    if dx < -0.002:
        _MODE["m"] = 1
        print(f"[smoke] force-frame probe: tray moved {dx * 1000:+.1f}mm -> mode 1",
              flush=True)
    print(f"[smoke] force-frame probe: dx={dx * 1000:+.1f}mm mode={_MODE['m']}",
          flush=True)
    if abs(dx) < 0.002:
        print("[smoke] WARNING: frame probe barely moved the tray", flush=True)

    # ================= 2. randomization is real (readback) ========================================
    def readback():
        _refresh()
        return (yaw_of(scene._st_quat[0]), scene._st_pos[0, :2].clone(),
                scene.station_local(scene.tray.data.root_pos_w)[0, :2].clone(),
                yaw_of(scene.tray.data.root_quat_w[0]))

    torch.manual_seed(101)
    env.reset()
    _step(10)
    a_yaw, a_sp, a_tp, a_ty = readback()
    torch.manual_seed(202)
    env.reset()
    _step(10)
    b_yaw, b_sp, b_tp, b_ty = readback()
    d_yaw, d_sp = dyaw(a_yaw, b_yaw), float((a_sp - b_sp).norm())
    d_tp, d_ty = float((a_tp - b_tp).norm()), dyaw(a_ty, b_ty)
    print(f"[smoke] randomization deltas: station_yaw={d_yaw:.1f}deg "
          f"station_xy={d_sp * 1000:.1f}mm tray_xy={d_tp * 1000:.1f}mm "
          f"tray_yaw={d_ty:.1f}deg", flush=True)
    check("randomization-is-real: station yaw, station xy and the tray start pose "
          "readback differ across seeds",
          d_yaw > 2.0 and d_sp > 0.003 and d_tp > 0.005 and d_ty > 2.0)

    # ================= 3. queue shuffle ===========================================================
    lvls_seen = set()
    for s in range(10):
        torch.manual_seed(300 + s)
        env.reset()
        _refresh()
        lvls_seen.add(int(scene.levels[0, 0]))
    print(f"[smoke] over 10 resets: ketchup queue levels {sorted(lvls_seen)}", flush=True)
    check("queue shuffle: the ketchup occupies >= 2 different queue levels over "
          "10 resets", len(lvls_seen) >= 2)

    # ================= 4. null policy fails =======================================================
    torch.manual_seed(100)
    env.reset()
    _step(20)
    t0 = scene.tray.data.root_pos_w[0].clone()
    _step(240)
    _report("null-policy")
    drift = float((scene.tray.data.root_pos_w[0] - t0).norm())
    in_mag = all(bool(scene.in_magazine(b)[0]) for b in scene.items)
    check("null-policy-fails: 240 idle steps — cartons stay queued, tray stays put "
          f"(moved {drift * 1000:.1f}mm < 8mm), score ~0, no success",
          in_mag and drift < 0.008 and float(scene.score()[0]) <= 0.01
          and not bool(scene.success()[0]))

    # ================= 5. SEED STRATEGY dead: the queue is captive ================================
    # The seed's plan starts with "pick up the ketchup". A 10 N upward yank on the
    # queued ketchup demonstrably lifts the queue (non-vacuous) but the ROOF holds
    # everything inside; an 8 N rearward pull is stopped by the rear wall (44 mm
    # slot < 50 mm carton). The carton never leaves the magazine.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _REC["on"] = True
    z0 = float(loc_of(scene.ketchup)[2])
    z_max = z0
    q_ref = scene.ketchup.data.root_quat_w.clone()
    f_up = torch.tensor([0.0, 0.0, 10.0], device=device)
    for _ in range(180):
        _force_on(scene.ketchup, f_up, q_ref)
        _step(1)
        z_max = max(z_max, float(loc_of(scene.ketchup)[2]))
    _force_off(scene.ketchup)
    _step(90)
    up_ok = bool(scene.in_magazine(scene.ketchup)[0])
    print(f"[smoke] captive-up: rose {(z_max - z0) * 1000:.1f}mm (roof holds), "
          f"in_magazine={up_ok}", flush=True)
    x_min = float(loc_of(scene.ketchup)[0])
    _hold_force(scene.ketchup, -8.0 * x_dir_w(), 180)
    _step(90)
    x_min = min(x_min, float(loc_of(scene.ketchup)[0]))
    rear_ok = bool(scene.in_magazine(scene.ketchup)[0])
    _report("captive-queue")
    check("SEED STRATEGY dead (captive queue): a 10 N up-yank lifts the queue "
          ">= 8mm yet the roof keeps the ketchup in the magazine; an 8 N rearward "
          "pull is stopped by the rear wall — the ketchup cannot be picked up",
          (z_max - z0) >= 0.008 and up_ok and rear_ok and x_min > -0.012
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))
    _REC["on"] = False

    # ================= 6. the lip holds a quasi-static nudge ======================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    bi = bottom_item()
    body = scene.items[bi]
    x0 = float(loc_of(body)[0])
    _hold_force(body, 1.5 * x_dir_w(), 300)
    _step(90)
    x1 = float(loc_of(body)[0])
    print(f"[smoke] lip nudge: bottom carton ({scene.ITEMS[bi]}) x {x0 * 1000:.0f} -> "
          f"{x1 * 1000:.0f}mm", flush=True)
    check("lip holds a nudge: 1.5 N quasi-static push for 2.5 s leaves the bottom "
          "carton behind the lip, in the magazine, score ~0 (check 7 ejects with "
          "the same encoding at solve force — the pair is non-vacuous)",
          bool(scene.in_magazine(body)[0]) and (x1 - x0) < 0.020
          and float(scene.score()[0]) <= 0.01 and not bool(scene.success()[0]))

    # ================= 7. one-per-push metering + follower retention ==============================
    # Bottom carton must be a DISTRACTOR (so the pit costs nothing we assert about
    # later); push it out with no tray staged: exactly it lands in the pit, the
    # other two re-seat in the magazine.
    s7 = _reset_with(lambda: int(scene.levels[0, 0]) != 0, 700)
    _step(90)
    bi = bottom_item()
    body = scene.items[bi]
    others = [b for j, b in enumerate(scene.items) if j != bi]
    print(f"[smoke] metering (seed {s7}): pushing out bottom carton "
          f"{scene.ITEMS[bi]}", flush=True)
    _REC["on"] = True
    pushed = _dispense_push(body)
    _step(360)
    _report("metering")
    _REC["on"] = False
    in_pit = bool(scene.in_pit(body)[0])
    others_in = all(bool(scene.in_magazine(b)[0]) for b in others)
    check("one-per-push metering: the solve-style push ejects EXACTLY the bottom "
          "carton into the pit; the dropped follower re-seats behind the lip and "
          "both other cartons stay in the magazine",
          pushed and in_pit and others_in and not bool(scene.success()[0]))

    # ================= 8. ORDER violation: serve before staging = unrecoverable ===================
    # Ketchup at the bottom of the queue, NO tray staged: pushing it out drops it
    # into the pit. Staging the tray afterwards cannot recover — no success.
    s8 = _reset_with(lambda: int(scene.levels[0, 0]) == 0, 800)
    _step(90)
    print(f"[smoke] order violation (seed {s8}): ketchup is the bottom carton", flush=True)
    _REC["on"] = True
    pushed = _dispense_push(scene.ketchup)
    _step(360)
    _report("order-pit")
    k_in_pit = bool(scene.in_pit(scene.ketchup)[0])
    stage_tray()
    _step(120)
    _report("order-staged")
    _REC["on"] = False
    check("ORDER violation: the ketchup dispensed with no tray staged falls into "
          "the REJECT PIT; staging the tray afterwards cannot help — no success, "
          "score stays at latched partial credit",
          pushed and k_in_pit and bool(scene.tray_seated()[0])
          and not bool(scene.in_tray(scene.ketchup)[0])
          and float(scene.score()[0]) <= 0.31 and not bool(scene.success()[0]))

    # ================= 9. wrong object ============================================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    stage_tray()
    _write_body(scene.items[1], tray_world((0.0, 0.0, 0.060)), scene._st_quat)
    _step(150)
    _report("wrong-object")
    butter_in = bool(scene.in_tray(scene.items[1])[0])
    s9a = float(scene.score()[0])
    ok9a = butter_in and not bool(scene.success()[0]) and s9a <= 0.31
    _write_body(scene.ketchup, tray_world((0.0, 0.0, 0.130)), scene._st_quat)
    _step(180)
    _report("wrong+ketchup")
    ok9b = not bool(scene.success()[0])
    check("wrong object: BUTTER in the seated tray is never success (score <= "
          "staged+reject credit); adding the ketchup on top still fails (dirty "
          "tray) — no prefix of the construction scores 1",
          ok9a and ok9b and float(scene.score()[0]) < 0.99)

    # ================= 10. unseated serve =========================================================
    torch.manual_seed(100)
    env.reset()
    _step(60)
    _write_body(scene.ketchup, tray_world((0.0, 0.0, 0.060)), scene._st_quat)
    _step(150)
    _report("unseated-serve")
    s10 = float(scene.score()[0])
    check("unseated serve: ketchup at rest in the tray ON THE GROUND — served "
          "credit only (~0.35), tray not on the rim, no success",
          bool(scene.in_tray(scene.ketchup)[0]) and not bool(scene.tray_seated()[0])
          and 0.34 <= s10 <= 0.36 and not bool(scene.success()[0]))

    # ================= 11. follower pile (dirty-gate regression) ==================================
    # The exact loophole the dirty gate exists for: ketchup served correctly AND a
    # follower piled on top of it. in_tray(ketchup) and tray_seated BOTH hold — the
    # only failing clause is the dirty gate.
    torch.manual_seed(100)
    env.reset()
    _step(60)
    stage_tray()
    _write_body(scene.ketchup, tray_world((0.0, 0.0, 0.060)), scene._st_quat)
    _step(120)
    _REC["on"] = True
    _write_body(scene.items[1], tray_world((0.0, 0.0, 0.150)), scene._st_quat)
    _step(180)
    _report("follower-pile")
    _REC["on"] = False
    k_in = bool(scene.in_tray(scene.ketchup)[0])
    seated = bool(scene.tray_seated()[0])
    dirty = bool(scene.over_tray(scene.items[1])[0])
    check("follower pile: ketchup in the SEATED tray with butter piled on top — "
          "in_tray(ketchup) and tray_seated hold, the dirty gate catches the "
          "pile, success is False",
          k_in and seated and dirty and not bool(scene.success()[0])
          and float(scene.score()[0]) < 0.99)

    # ================= save + verdict =============================================================
    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.queue_dispenser")
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
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print("SIM_GEN_SMOKE: FAIL (exception)", flush=True)
        os._exit(2)

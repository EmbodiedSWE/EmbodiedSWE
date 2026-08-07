"""Smoke battery for GumballMeterScene (sim_gen task
`libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_i34`)
— REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS a settled wrong outcome and asserts the rubric refuses it. success()
must never fire anywhere in the battery.

 1. settle/no-NaN     — shuttle parked IN, pocket loaded, basin empty, score ~0.
 2. randomization A   — tower yaw spans > 90 deg and xy jitters across resets.
 3. randomization B   — quota K and stock both vary across resets, and the number
                        of standing green markers equals K every time (readback).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (pull the sliding part out, push it
                        back in — no counting) executed for real ONCE: exactly one
                        ball dispensed, score capped at one cycle's credit.
 6. one-per-cycle     — the shuttle held at the OUT stop for ~5 s: its solid rear
                        seals the silo, so the basin gains at most this cycle's
                        single ball — an open "gate" does NOT stream balls.
 7. overfill spoiled  — K+1 ambers in the basin (constructed, settled): the
                        overfill latch fires, score clamps to the spoil cap.
 8. spoil irreversible— one overfill ball teleported back out (count == K again,
                        everything else perfect): success STILL refused.
 9. under-count       — K-1 ambers in the basin: no success, partial credit only.
10. exact but OUT     — K ambers in the basin but the shuttle left at the OUT
                        stop: parked-IN clause refuses.
11. red decoy         — K ambers correct PLUS the red ball inside the basin:
                        refused.
12. stray amber       — K ambers in the basin but another ACTIVE amber loose on
                        the floor: the every-ball-accounted-for clause refuses.
13. latched credit    — one dispensed ball (constructed) then removed: the count
                        latch survives, success does not.
14. rejection audit   — success() observed False at every step of the battery.
15. final no-NaN      — and frames.npz written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer_i34.smoke
             --headless
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

# Global watchdog: if anything wedges, die loudly before the forge timeout.
threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                 os._exit(3))).start()

# ----- module state wired up in main() ------------------------------------------------------------
_ENV = None
_REC = {"on": True, "annot": None, "frames": [], "i": 0}
_AUDIT = {"saw_success": False}


def _step(k: int = 1) -> None:
    env = _ENV
    no_action = torch.empty(0, device=env.device)
    for _ in range(k):
        rec = _REC["on"] and _REC["annot"] is not None
        env.step(no_action, render=rec)
        _AUDIT["saw_success"] |= bool(env.scene.success()[0])
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
    print(f"[smoke] {tag:18s} | shut_x={float(scene.shuttle_x()[0]):+.3f} "
          f"parked_in={bool(scene.shuttle_parked_in()[0])} "
          f"K={int(scene.k_target[0])} stock={int(scene.stock[0])} "
          f"basin={int(scene.basin_count()[0])} "
          f"stowed={int((scene.balls_stowed() & scene.active)[0].sum())} "
          f"over={bool(scene._over[0])} maxcnt={int(scene._maxcnt[0])} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.gumball_meter")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.85, -0.85, 0.70)) + o),
                                tuple(np.array((0.02, 0.02, 0.15)) + o),
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
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video",
              flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def succ() -> bool:
        s = bool(scene.success()[0])
        _AUDIT["saw_success"] |= s
        return s

    def yaw_of(quat_row: torch.Tensor) -> float:
        return math.degrees(2.0 * math.atan2(float(quat_row[3]), float(quat_row[0])))

    def tower_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.tower.data.root_pos_w + quat_apply(
            scene.tower.data.root_quat_w, loc)

    # basin drop slots (tower-local), >= 30 mm apart: 2x2 floor grid + one ramp spot
    SLOTS = [(-0.004, -0.016), (-0.004, +0.016), (0.026, -0.016),
             (0.026, +0.016), (0.056, 0.000)]

    def put_in_basin(ball: str, slot: int) -> None:
        """CONSTRUCT: write the amber ball just above the basin floor; it settles
        inside (the ONLY legit route in is the drop hole — this is a rubric probe)."""
        x, y = SLOTS[slot]
        _write_body(scene.balls[ball], tower_world([x, y, 0.045]))

    def put_stowed(ball: str) -> None:
        """CONSTRUCT: write the amber ball high in the silo bore; it falls onto the
        pocket/stack and counts as stowed."""
        _write_body(scene.balls[ball], tower_world([0.0, 0.0, 0.300]))

    def park_shuttle(x_local: float) -> None:
        """CONSTRUCT: write the shuttle parked at `x_local` in the channel."""
        _write_body(scene.shuttle, tower_world([x_local, 0.0, c.shut_z]),
                    scene.tower.data.root_quat_w)

    def tail_axis() -> torch.Tensor:
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        return quat_apply(scene.tower.data.root_quat_w, ex)

    def drive(target_x: float, clamp: float, kd: float, steps: int) -> bool:
        """REAL actuation: PD force on the knob along the tail axis (what a hand
        does); the channel's end walls arrest the shuttle."""
        zero = torch.zeros(n, 1, 3, device=device)
        ok_frames = 0
        reached = False
        for _ in range(steps):
            axis = tail_axis()
            v = (scene.shuttle.data.root_lin_vel_w * axis).sum(-1)
            f_mag = (200.0 * (target_x - scene.shuttle_x()) - kd * v).clamp(-clamp, clamp)
            fw = axis * f_mag.unsqueeze(-1)
            fb = quat_apply_inverse(scene.shuttle.data.root_quat_w, fw)
            scene.shuttle.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            _step(1)
            near = abs(float(scene.shuttle_x()[0]) - target_x) < 0.006 \
                and abs(float(v[0])) < 0.02
            ok_frames = ok_frames + 1 if near else 0
            if ok_frames >= 5:
                reached = True
                break
        scene.shuttle.set_external_force_and_torque(zero, zero)
        _step(20)
        return reached

    def wiggle() -> None:
        zero = torch.zeros(n, 1, 3, device=device)
        for i in range(8):
            sgn = 1.0 if i % 2 == 0 else -1.0
            for _ in range(5):
                axis = tail_axis()
                fb = quat_apply_inverse(scene.shuttle.data.root_quat_w, axis * (sgn * 8.0))
                scene.shuttle.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
                _step(1)
        scene.shuttle.set_external_force_and_torque(zero, zero)
        _step(15)

    def seek(target_x: float, clamp: float, kd: float) -> bool:
        for _ in range(4):
            if drive(target_x, clamp=clamp, kd=kd, steps=300):
                return True
            wiggle()
        return abs(float(scene.shuttle_x()[0]) - target_x) < 0.010

    def pull_out() -> bool:
        return seek(0.040, clamp=10.0, kd=30.0) \
            and seek(c.travel + 0.002, clamp=3.0, kd=60.0)

    def push_in() -> bool:
        return seek(0.030, clamp=8.0, kd=25.0) \
            and seek(-0.002, clamp=4.0, kd=60.0)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(120)
    _report("reset")
    check("settle/no-NaN: shuttle parked IN, basin empty, all stock stowed in the "
          "silo column, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.shuttle_parked_in()[0])
          and int(scene.basin_count()[0]) == 0
          and int((scene.balls_stowed() & scene.active)[0].sum())
          == int(scene.stock[0])
          and float(scene.score()[0]) <= 0.02 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, ks, stocks, match = [], [], [], [], True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(6)
        _refresh()
        yaws.append(yaw_of(scene.tower.data.root_quat_w[0]))
        xys.append((scene.tower.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        ks.append(int(scene.k_target[0]))
        stocks.append(int(scene.stock[0]))
        shown = sum(1 for m in scene.markers.values()
                    if float(m.data.root_pos_w[0, 2]) > -0.4)
        match = match and (shown == ks[-1])
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} K={ks} stock={stocks} markers_match={match}", flush=True)
    check("randomization A: tower yaw spans > 90 deg and xy jitters across resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: quota K and stock both vary, and the standing-marker "
          "count equals K on every reset",
          len(set(ks)) >= 2 and len(set(stocks)) >= 2 and match)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.02 and not succ())

    # ================= 5. SEED strategy (one real uncounted cycle) ================================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    K5 = int(scene.k_target[0])
    ok_pull = pull_out()
    for _ in range(720):
        _step(1)
        if int(scene.basin_count()[0]) >= 1 and bool(scene.settled()[0]):
            break
    ok_push = push_in()
    _step(60)
    _report("seed-skill")
    s5 = float(scene.score()[0])
    cap5 = c.w_pull + c.w_count / K5 + 1e-6
    check("SEED strategy: the seed's whole skill (pull the sliding part, push it "
          "back — no counting) executed for real once: exactly 1 ball dispensed, "
          f"score <= one cycle's credit ({cap5:.2f}), no success",
          ok_pull and ok_push and int(scene.basin_count()[0]) == 1
          and 0.05 <= s5 <= cap5 and not succ())

    # ================= 6. one-per-cycle (held-OUT does not stream) ================================
    ok_pull6 = pull_out()
    _step(600)   # ~5 s with the shuttle parked at the OUT stop
    _report("held-out")
    check("one-per-cycle: shuttle held at the OUT stop for ~5 s — its solid rear "
          "seals the silo, so the basin holds exactly 2 balls (cycles 1+2), the "
          "rest of the stock stays stowed, no success",
          ok_pull6 and float(scene.shuttle_x()[0]) > c.travel - 0.010
          and int(scene.basin_count()[0]) == 2
          and int((scene.balls_stowed() & scene.active)[0].sum())
          == int(scene.stock[0]) - 2 and not succ())

    # ================= 7. overfill spoiled ========================================================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    K7 = int(scene.k_target[0])
    for j in range(K7 + 1):                     # ALL writes before any step: the
        put_in_basin(f"amber_{j}", j)           # count jumps 0 -> K+1 directly
    _step(240)
    _report("overfill")
    check("overfill spoiled: K+1 ambers in the sealed basin (constructed, settled) "
          "— overfill latch fires, score clamps to the spoil cap, no success",
          bool(scene._over[0]) and int(scene.basin_count()[0]) == K7 + 1
          and float(scene.score()[0]) <= c.spoil_cap + 1e-6 and not succ())

    # ================= 8. spoil is irreversible ===================================================
    put_stowed("amber_0")                       # teleport one ball back out (a hand
    _step(240)                                  # never could: the basin is sealed)
    _report("spoil-latch")
    check("spoil irreversible: one overfill ball removed, count == K again and "
          "everything else perfect — success STILL refused, score stays capped",
          int(scene.basin_count()[0]) == K7 and bool(scene.shuttle_parked_in()[0])
          and bool(scene._over[0]) and float(scene.score()[0]) <= c.spoil_cap + 1e-6
          and not succ())

    # ================= 9. under-count =============================================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    K9 = int(scene.k_target[0])
    for j in range(1, K9):                      # K-1 balls from the silo stack
        put_in_basin(f"amber_{j}", j - 1)
    _step(240)
    _report("under-count")
    s9 = float(scene.score()[0])
    check("under-count: K-1 ambers in the basin — no success, partial credit only",
          int(scene.basin_count()[0]) == K9 - 1 and not succ()
          and 0.20 <= s9 <= 0.55)

    # ================= 10. exact count but shuttle OUT ============================================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    K10 = int(scene.k_target[0])
    park_shuttle(c.travel)                      # spoiler FIRST: never a success state
    for j in range(K10):
        put_in_basin(f"amber_{j}", j)
    _step(240)
    _report("exact-but-OUT")
    check("exact but OUT: K ambers in the basin but the shuttle left at the OUT "
          "stop — the parked-IN clause refuses success",
          int(scene.basin_count()[0]) == K10
          and not bool(scene.shuttle_parked_in()[0]) and not succ())

    # ================= 11. red decoy in the basin =================================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    K11 = int(scene.k_target[0])
    _write_body(scene.decoy, tower_world([0.056, 0.000, 0.050]))   # decoy FIRST
    for j in range(1, K11 + 1):                 # ambers from the silo stack
        put_in_basin(f"amber_{j}", j - 1)
    _step(240)
    _report("red-decoy")
    check("red decoy: exact amber count PLUS the red ball inside the basin — "
          "refused",
          int(scene.basin_count()[0]) == K11 and bool(scene.decoy_in_basin()[0])
          and not succ())

    # ================= 12. stray amber ============================================================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    K12 = int(scene.k_target[0])
    ground = (scene.env_origins[0:1] + torch.tensor(
        [0.85, 0.85, c.ball_r + 0.001], device=device)).expand(n, 3)
    _write_body(scene.balls["amber_0"], ground)   # stray FIRST (from the pocket)
    for j in range(1, K12 + 1):
        put_in_basin(f"amber_{j}", j - 1)
    _step(240)
    _report("stray-amber")
    check("stray amber: K ambers in the basin but another ACTIVE amber loose on "
          "the floor — the every-ball-accounted-for clause refuses",
          int(scene.basin_count()[0]) == K12 and not succ())

    # ================= 13. latched credit =========================================================
    torch.manual_seed(101)
    env.reset()
    _step(120)
    K13 = int(scene.k_target[0])
    put_in_basin("amber_1", 1)
    _step(180)
    latched = int(scene._maxcnt[0]) >= 1
    _write_body(scene.balls["amber_1"], ground)   # remove it again
    _step(120)
    _report("latch")
    s13 = float(scene.score()[0])
    lo, hi = c.w_count / 3 - 0.02, c.w_count / 2 + 0.02
    check("latched credit: one dispensed ball (constructed) then removed — the "
          "count latch survives, success does not",
          latched and int(scene._maxcnt[0]) >= 1
          and int(scene.basin_count()[0]) == 0 and lo <= s13 <= hi and not succ())

    # ================= 14 + 15. audit, video, verdict =============================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.gumball_meter")
        print(f"[smoke] wrote {args.out}: {arr.shape}", flush=True)
    check("video: >10 frames recorded", len(_REC["frames"]) > 10)

    n_pass = sum(1 for _, ok in checks if ok)
    if n_pass == len(checks):
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
        code = 0
    else:
        for nm, ok in checks:
            if not ok:
                print(f"[smoke] FAILED: {nm}", flush=True)
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)}", flush=True)
        code = 1
    # Hard exit: Kit teardown hangs — watchdog then die.
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

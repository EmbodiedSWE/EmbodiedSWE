"""Smoke battery for BoltLatchStowScene (sim_gen task
`libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i420`)
— REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS (or physically drives) a wrong outcome and asserts the rubric
refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — drawer at its bolt-latched rest, bolt seated, bowl and
                        bottle on the bench, score ~0, no success.
 2. randomization A   — the bowl's bench xy varies across resets (readback).
 3. randomization B   — bowl and bottle always spawn on OPPOSITE sides and the
                        bowl's side flips across resets.
 4. null policy       — 240 idle steps -> everything holds still, score ~0.
 5. SEED strategy     — the seed's opening skill (pull the drawer open from the
                        front) executed for REAL: a 25 N PD pull (the spring
                        even helps it) presses the panel into the bolt's flat
                        rear face — the drawer never opens more than a few mm;
                        score ~0, no success.
 6. seed end state    — the seed's TERMINAL state constructed: drawer open at
                        its stop, bowl settled inside the OPEN drawer. Capped
                        partial credit only; 240 settled steps, never success.
 7. spring reopen     — bowl + drawer constructed near-shut (~9 cm) but
                        UNLATCHED, zero velocity: the spring throws the drawer
                        back to its stop; shut() never holds, no success.
 8. wrong object      — the FULL real cycle with the wrong payload: bolt lifted
                        by real force (spring opens), the green BOTTLE laid
                        into the drawer, drawer pushed shut by real force and
                        RE-LATCHED through the real cam. The latch holds
                        hands-off — and success never fires (bowl on bench);
                        score <= release+open credit.
 9. give-up push      — real release, then a real push that stops ~3 cm short
                        of the latch and lets go: the spring reopens the drawer
                        to its stop; the shut band is never entered; no success.
10. settle gate       — bowl written INSIDE the shut cavity WITH velocity: the
                        settle gate refuses success on the moving state
                        (checked without stepping; then retracted).
11. latched credit    — real release (release+open latches earn 0.35), then the
                        drawer stolen back shut by teleport: latches survive,
                        but a shut EMPTY drawer is not success.
12. rejection audit   — success() observed False at every step of the battery.
13. final no-NaN.
14. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_i420.smoke --headless
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
_wd = threading.Timer(1350.0, lambda: (print("SIM_GEN_SMOKE: FAIL (watchdog)", flush=True),
                                       os._exit(3)))
_wd.daemon = True
_wd.start()

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


def _write_body(body, pos_w: torch.Tensor, quat=None, vel_x: float = 0.0) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    if quat is None:
        st[:, 3] = 1.0
    else:
        st[:, 3:7] = torch.tensor(quat, device=_ENV.device)
    st[:, 7] = vel_x
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:18s} | open={float(scene.opening()[0]):+.4f} "
          f"lift={float(scene.bolt_lift()[0]):+.4f} "
          f"in_cav={bool(scene.bowl_in_cavity()[0])} shut={bool(scene.shut()[0])} "
          f"latch(r/o/l)=({int(scene._released[0])},{int(scene._opened[0])},"
          f"{int(scene._loaded[0])}) settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.boltlatch_stow")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.30, -0.95, 0.85)) + o),
                                tuple(np.array((-0.02, 0.00, 0.22)) + o),
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

    def origin(dx: float, dy: float, dz: float) -> torch.Tensor:
        return (scene.env_origins[0:1]
                + torch.tensor([dx, dy, dz], device=device)).expand(n, 3)

    zero = torch.zeros(n, 1, 3, device=device)

    def body_force(body, f_w: torch.Tensor) -> None:
        f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
        body.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)

    def pull_drawer(steps: int, *, kp: float, kd: float, clamp: float) -> float:
        """REAL actuation: outward PD pull on the drawer toward past-the-stop;
        returns the PEAK opening seen during the pull (measured live — gravity/
        spring restore afterwards)."""
        peak = 0.0
        for _ in range(steps):
            d = scene.opening()
            v = scene.drawer.data.root_lin_vel_w[:, 0]
            fx = (kp * (c.stroke + 0.02 - d) - kd * v).clamp(min=-clamp, max=clamp)
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = fx
            body_force(scene.drawer, f_w)
            _step(1)
            peak = max(peak, float(scene.opening()[0]))
        scene.drawer.set_external_force_and_torque(zero, zero)
        _step(60)
        return peak

    def lift_bolt_until_open(max_steps: int = 480) -> None:
        """REAL release: T-handle lift servo (force-limited) until the spring
        has thrown the drawer past `opened_min`; then hands off the bolt."""
        for _ in range(max_steps):
            z = scene.bolt_lift()
            vz = scene.bolt.data.root_lin_vel_w[:, 2]
            fz = (4.9 + 30.0 * (0.035 - z) - 4.0 * vz).clamp(min=-3.0, max=9.0)
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 2] = fz
            body_force(scene.bolt, f_w)
            _step(1)
            if float(scene.opening()[0]) >= c.opened_min + 0.01:
                break
        scene.bolt.set_external_force_and_torque(zero, zero)
        _step(180)

    def close_drawer(*, done, steps: int = 720, v_des: float = 0.15,
                     ff: float = 7.0, clamp: float = 14.0) -> float:
        """REAL actuation: velocity-limited palm-push (world -x) against the
        live spring; returns the MINIMUM opening seen. Forces off afterwards."""
        lo = float(scene.opening()[0])
        for _ in range(steps):
            vx = scene.drawer.data.root_lin_vel_w[:, 0]
            fx = (-ff + 60.0 * (-v_des - vx)).clamp(min=-clamp, max=2.0)
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = fx
            body_force(scene.drawer, f_w)
            _step(1)
            lo = min(lo, float(scene.opening()[0]))
            if done():
                break
        scene.drawer.set_external_force_and_torque(zero, zero)
        return lo

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    check("settle/no-NaN: drawer at its bolt-latched rest, bolt seated, "
          "score ~0, no success",
          bool(scene._finite()[0])
          and 0.003 <= float(scene.opening()[0]) <= c.shut_tol
          and float(scene.bolt_lift()[0]) < 0.004
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    bxys, sides_ok, bowl_sides = [], [], []
    for k in range(6):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        bw = (scene.bowl.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist()
        bt = (scene.bottle.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist()
        bxys.append(bw)
        sides_ok.append(bw[1] * bt[1] < 0.0)
        bowl_sides.append(bw[1] > 0.0)
    bxy_std = float(np.std(np.asarray(bxys), axis=0).mean())
    print(f"[smoke] readback: bowl_xy={[f'({x:+.3f},{y:+.3f})' for x, y in bxys]} "
          f"std={bxy_std:.4f} opposite={sides_ok} sides={bowl_sides}", flush=True)
    check("randomization A: the bowl's bench xy varies across resets",
          bxy_std > 0.008)
    check("randomization B: bowl and bottle always on OPPOSITE sides; the "
          "bowl's side flips across resets",
          all(sides_ok) and (True in bowl_sides) and (False in bowl_sides))

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> drawer stays latched shut, score ~0, "
          "no success",
          float(scene.opening()[0]) <= c.shut_tol
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (real pull on the latched drawer) ========================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    peak5 = pull_drawer(300, kp=400.0, kd=30.0, clamp=25.0)
    _report("seed-pull")
    check("SEED strategy: a real 25 N pull (spring helping) cannot open the "
          "bolt-latched drawer — peak opening a few mm, score ~0, no success",
          peak5 <= 0.030
          and float(scene.opening()[0]) <= c.shut_tol + 0.002
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. seed END STATE (bowl in the OPEN drawer) ================================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    _write_body(scene.drawer, origin(c.stroke, 0.0, 0.0))
    _write_body(scene.bowl, origin(c.drop_x(), 0.0, c.floor_z1 + c.bowl_h / 2 + 0.003))
    _step(240)
    _report("seed-end")
    s6 = float(scene.score()[0])
    check("seed end state: bowl settled inside the OPEN drawer (the seed's "
          "terminal state) — capped partial credit, 240 settled steps, never "
          "success",
          bool(scene.bowl_in_cavity()[0])
          and float(scene.opening()[0]) >= c.stroke - 0.02
          and bool(scene.settled()[0])
          and s6 <= 0.60 + 1e-4 and not succ())

    # ================= 7. spring reopen (near-shut but unlatched) =================================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    _write_body(scene.drawer, origin(0.090, 0.0, 0.0))
    _write_body(scene.bowl, origin(-0.010, 0.0, c.floor_z1 + c.bowl_h / 2 + 0.003))
    never_shut = True
    for _ in range(300):
        _step(1)
        never_shut = never_shut and not bool(scene.shut()[0])
    _report("spring-reopen")
    check("spring reopen: drawer constructed near-shut (~9 cm) but UNLATCHED "
          "with the bowl inside — the spring throws it back to its stop; the "
          "shut band is never entered, no success",
          never_shut
          and float(scene.opening()[0]) >= c.stroke - 0.02
          and not succ())

    # ================= 8. wrong object (full REAL cycle with the bottle) ==========================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    lift_bolt_until_open()
    opened8 = float(scene.opening()[0]) >= c.stroke - 0.02
    # lay the green bottle into the open drawer (side-lying, axis along y)
    _write_body(scene.bottle, origin(c.drop_x(), 0.0, c.floor_z1 + c.bottle_r + 0.004),
                quat=(0.7071068, 0.7071068, 0.0, 0.0))
    _step(90)
    close_drawer(done=lambda: float(scene.opening()[0]) <= 0.004)
    for _ in range(50):
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 0] = -8.0
        body_force(scene.drawer, f_w)
        _step(1)
    scene.drawer.set_external_force_and_torque(zero, zero)
    _step(240)
    _report("wrong-object")
    s8 = float(scene.score()[0])
    check("wrong object: the full REAL cycle (bolt lifted, spring opened, "
          "BOTTLE laid in, drawer pushed shut and re-latched through the real "
          "cam) — the latch holds hands-off, but success never fires and score "
          "stays at the release+open credit",
          opened8
          and float(scene.opening()[0]) <= c.shut_tol
          and float(scene.bolt_lift()[0]) < 0.005
          and not bool(scene.bowl_in_cavity()[0])
          and s8 <= c.w_release + c.w_open + 1e-4 and not succ())

    # ================= 9. give-up push (stops short of the latch) =================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    lift_bolt_until_open()
    lo9 = close_drawer(done=lambda: float(scene.opening()[0]) <= 0.032,
                       v_des=0.10, clamp=12.0)
    _step(300)
    _report("give-up")
    check("give-up push: a real push that stops ~3 cm short of the latch and "
          "lets go — the shut band is never entered and the spring reopens the "
          "drawer to its stop; no success",
          lo9 > c.shut_tol
          and float(scene.opening()[0]) >= c.stroke - 0.02
          and not succ())

    # ================= 10. settle gate (moving bowl inside the shut cavity) =======================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    _write_body(scene.bowl, origin(-0.100, 0.0, c.floor_z1 + c.bowl_h / 2 + 0.002),
                vel_x=0.40)
    in_shut_cavity = bool(scene.bowl_in_cavity()[0]) and bool(scene.shut()[0])
    moving_rejected = not succ()  # judged WITHOUT stepping: in place but moving
    # retract before any stepping — letting this settle would BE the goal state
    _write_body(scene.bowl, origin(float(scene.bowl_xy0[0, 0]),
                                   float(scene.bowl_xy0[0, 1]),
                                   c.bench_z1 + c.bowl_h / 2 + 0.002))
    _step(90)
    _report("settle-gate")
    check("settle gate: bowl written INSIDE the shut cavity WITH velocity — "
          "the moving state is refused (then retracted without stepping)",
          in_shut_cavity and moving_rejected and not succ())

    # ================= 11. latched credit (real release, drawer stolen shut) ======================
    torch.manual_seed(101)
    env.reset()
    _step(120)
    lift_bolt_until_open()
    latched11 = bool(scene._released[0]) and bool(scene._opened[0])
    _write_body(scene.drawer, origin(0.0, 0.0, 0.0))  # steal it back shut
    _step(120)
    _report("latch-theft")
    s11 = float(scene.score()[0])
    check("latched credit: real release earned release+open (0.35, latches "
          "survive the theft), but a shut EMPTY drawer is not success",
          latched11
          and float(scene.opening()[0]) <= c.shut_tol
          and c.w_release + c.w_open - 1e-4 <= s11 <= 0.60 + 1e-4
          and not succ())

    # ================= 12-14. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.boltlatch_stow")
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
    except Exception as e:  # noqa: BLE001 - fast fail beats a 20-min watchdog hang
        import traceback

        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

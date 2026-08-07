"""Smoke battery for RockerCabinetScene (sim_gen task
`libero_kitchen_scene1_open_bottom_drawer_i87`) — REJECTION-ONLY: every check
either verifies basic health/randomization or CONSTRUCTS (or physically drives)
a settled wrong outcome and asserts the rubric refuses it. success() must never
fire anywhere in the battery.

 1. settle/no-NaN     — top drawer resting at its sampled opening, bottom drawer
                        sealed at 0, score ~0, no success.
 2. randomization A   — the top drawer's initial opening d0 varies across resets
                        and the settled drawer tracks it every time (readback).
 3. randomization B   — bowl and plate xy poses vary across resets (readback).
 4. null policy       — 240 idle steps -> nothing moves, score ~0, no success.
 5. SEED strategy     — the seed's whole skill (pull a drawer open from its
                        front) executed for REAL: a 20 N PD pull drags the TOP
                        drawer to its outer stop (asserted: it actually moved).
                        The rocker just disengages; the bottom drawer stays
                        sealed; score stays ~0, no success.
 6. bottom push probe — REAL force on the judged drawer: first constructed 2 cm
                        open, a 15 N inward push drives it back to its inner
                        stop (asserted: it actually moved — the force pathway is
                        live), then keeps pressing: the drawer cannot be pushed
                        open and stays at 0; no success. (Pulling it open is
                        geometrically impossible — no purchase — and an external
                        pull force would bypass that geometry, so it is not a
                        valid probe; see TASK.md.)
 7. timid push        — a REAL inward push on the top drawer that stops ~2 cm in
                        (asserted moved): below the engage latch, transmission
                        barely cracks a few mm; score ~0, no success.
 8. near miss         — bottom drawer constructed settled at ~7 cm (past both
                        partial latches, short of the 8 cm goal): capped partial
                        credit only, no success.
 9. open but moving   — bottom drawer written at 10 cm WITH outward velocity:
                        the settle gate refuses success on the moving state
                        (checked without stepping; then retracted).
10. wrong object      — bowl and plate shoved off the cabinet to the ground:
                        score ~0, no success.
11. latched credit    — the REAL mechanism driven partway (top pushed until the
                        bottom cracks ~3.5 cm, force released), then the bottom
                        drawer stolen back shut: engage+crack latches survive,
                        success does not.
12. rejection audit   — success() observed False at every step of the battery.
13. final no-NaN.
14. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene1_open_bottom_drawer_i87.smoke --headless
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


def _write_body(body, pos_w: torch.Tensor, vel_x: float = 0.0) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    st[:, 3] = 1.0
    st[:, 7] = vel_x
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    print(f"[smoke] {tag:18s} | top={float(scene.top_open()[0]):+.4f} "
          f"d0={float(scene.d0[0]):+.4f} bot={float(scene.bot_open()[0]):+.4f} "
          f"latch(e/c/a)=({int(scene._engage[0])},{int(scene._crack[0])},"
          f"{int(scene._ajar[0])}) settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rocker_cabinet")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.25, -1.05, 0.95)) + o),
                                tuple(np.array((-0.20, 0.00, 0.30)) + o),
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

    def push_drawer(body, opening_fn, tgt: float, steps: int, *, kp: float,
                    kd: float, clamp: float) -> None:
        """REAL actuation: a PD force along the drawer's prismatic axis toward
        opening `tgt`, force-limited to what a hand would apply."""
        for _ in range(steps):
            d = opening_fn()
            v = body.data.root_lin_vel_w[:, 0]
            fx = (kp * (tgt - d) - kd * v).clamp(min=-clamp, max=clamp)
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = fx
            f_b = quat_apply_inverse(body.data.root_quat_w, f_w)
            body.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
        body.set_external_force_and_torque(zero, zero)
        _step(60)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    check("settle/no-NaN: top drawer resting at its sampled opening, bottom "
          "drawer sealed at 0, score ~0, no success",
          bool(scene._finite()[0])
          and abs(float(scene.top_open()[0]) - float(scene.d0[0])) < 0.006
          and float(scene.bot_open()[0]) < 0.005
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    d0s, tops, xys = [], [], []
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        d0s.append(float(scene.d0[0]))
        tops.append(float(scene.top_open()[0]))
        bw = (scene.bowl.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist()
        pl = (scene.plate.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist()
        xys.append(bw + pl)
    d0_span = max(d0s) - min(d0s)
    tracks = all(abs(t - d) < 0.006 for t, d in zip(tops, d0s))
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: d0={[f'{d:.4f}' for d in d0s]} span={d0_span:.4f} "
          f"tracks={tracks} topper_xystd={xystd:.3f}", flush=True)
    check("randomization A: the top drawer's initial opening d0 varies across "
          "resets and the settled drawer tracks it every time",
          d0_span > 0.003 and tracks)
    check("randomization B: bowl and plate xy poses vary across resets",
          xystd > 0.008)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> bottom stays sealed, score ~0, no success",
          float(scene.bot_open()[0]) < 0.005
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (real pull on the top drawer) =============================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    top_before = float(scene.top_open()[0])
    push_drawer(scene.top, scene.top_open, c.top_stroke + 0.02, 300,
                kp=300.0, kd=30.0, clamp=20.0)
    _report("seed-skill")
    top_after = float(scene.top_open()[0])
    check("SEED strategy: a real 20 N pull drags the top drawer OUT to its outer "
          "stop (it moved) — the rocker disengages, the bottom drawer stays "
          "sealed, score ~0, no success",
          top_after - top_before >= 0.006
          and top_after <= c.top_stroke + 0.005
          and float(scene.bot_open()[0]) < 0.005
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. bottom push probe (real force on the judged drawer) ====================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    # construct the bottom drawer 2 cm open (below every latch), settled
    _write_body(scene.bot, origin(0.020, 0.0, 0.0))
    _step(60)
    b_before = float(scene.bot_open()[0])
    # real inward push: it slides back to its inner stop — the force pathway is live
    push_drawer(scene.bot, scene.bot_open, -0.05, 240, kp=300.0, kd=30.0, clamp=15.0)
    b_mid = float(scene.bot_open()[0])
    # keep pressing against the stop: the drawer cannot be pushed open
    push_drawer(scene.bot, scene.bot_open, -0.05, 120, kp=300.0, kd=30.0, clamp=15.0)
    _report("bottom-push")
    check("bottom push probe: a real 15 N push on the judged drawer drives it "
          "back to its inner stop (it moved — the force is live) and pressing "
          "harder cannot open it; score ~0, no success",
          b_before >= 0.014 and b_before - b_mid >= 0.012
          and float(scene.bot_open()[0]) < 0.004
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 7. timid push (stops short of the engage latch) ============================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    d0_7 = float(scene.d0[0])
    push_drawer(scene.top, scene.top_open, d0_7 - 0.018, 240,
                kp=250.0, kd=40.0, clamp=10.0)
    _report("timid-push")
    moved_in = d0_7 - float(scene.top_open()[0])
    check("timid push: a real inward push that stops ~2 cm in (it moved) stays "
          "below the engage latch and barely cracks the bottom drawer; score ~0, "
          "no success",
          0.010 <= moved_in <= 0.028
          and not bool(scene._engage[0])
          and float(scene.bot_open()[0]) < c.crack_open
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. near miss (7 cm settled, goal is 8 cm) ==================================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    _write_body(scene.bot, origin(0.070, 0.0, 0.0))
    _step(120)
    _report("near-miss")
    s8 = float(scene.score()[0])
    check("near miss: bottom drawer constructed settled at ~7 cm — past both "
          "partial latches but short of the 8 cm goal: capped partial credit, "
          "no success",
          0.060 <= float(scene.bot_open()[0]) < c.open_goal
          and s8 <= 0.65 + 1e-4 and not succ())

    # ================= 9. open but moving (settle gate) ===========================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    _write_body(scene.bot, origin(0.100, 0.0, 0.0), vel_x=0.30)
    moving_rejected = not succ()  # judged WITHOUT stepping: open enough but moving
    still_open = float(scene.bot_open()[0]) >= c.open_goal
    _write_body(scene.bot, origin(0.020, 0.0, 0.0))  # retract before any stepping
    _step(90)
    _report("open-moving")
    check("open but moving: bottom drawer written at 10 cm with outward velocity "
          "— the settle gate refuses success on the moving state",
          moving_rejected and still_open and not succ())

    # ================= 10. wrong object (distractors shoved off) ==================================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    _write_body(scene.bowl, origin(0.60, 0.55, c.bowl_h / 2 + 0.002))
    _write_body(scene.plate, origin(0.60, -0.55, c.plate_h / 2 + 0.002))
    _step(180)
    _report("wrong-object")
    check("wrong object: bowl and plate shoved off the cabinet to the ground — "
          "score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 11. latched credit (real partial transmission, then stolen) ================
    torch.manual_seed(101)
    env.reset()
    _step(120)
    # drive the REAL mechanism partway: push the top drawer until the bottom
    # drawer cracks ~3.5 cm (nothing ever touches the bottom drawer)
    no_action = torch.empty(0, device=device)
    for _ in range(600):
        d = scene.top_open()
        v = scene.top.data.root_lin_vel_w[:, 0]
        fx = (400.0 * (0.0 - d) - 40.0 * v).clamp(min=-35.0, max=35.0)
        f_w = torch.zeros(n, 3, device=device)
        f_w[:, 0] = fx
        f_b = quat_apply_inverse(scene.top.data.root_quat_w, f_w)
        scene.top.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
        _step(1)
        if float(scene.bot_open()[0]) >= 0.035:
            break
    scene.top.set_external_force_and_torque(zero, zero)
    # sample AT the break — the coasting mechanism would otherwise drive the
    # bottom drawer all the way out (momentum run-away is the honest solve's
    # finishing move; here we must freeze the partial state)
    latched = bool(scene._engage[0]) and bool(scene._crack[0])
    b11 = float(scene.bot_open()[0])
    # arrest + steal: top drawer written back OUT to its sampled opening
    # (disengaged), rocker written back to rest (its swung lower pad would
    # otherwise depenetrate the shut plate right back out), bottom stolen shut
    _write_body(scene.top, origin(float(scene.d0[0]), 0.0, 0.0))
    _write_body(scene.rocker, origin(c.pivot_x, 0.0, c.pivot_z))
    _write_body(scene.bot, origin(0.0, 0.0, 0.0))
    _step(90)
    _report("latch")
    s11 = float(scene.score()[0])
    check("latched credit: the real transmission driven until the bottom cracks "
          "(engage+crack latch), then the bottom drawer stolen back shut — "
          "latches survive, success does not",
          latched and 0.030 <= b11 <= 0.062
          and float(scene.bot_open()[0]) < 0.005
          and c.w_engage + c.w_crack - 1e-4 <= s11 <= 0.65 + 1e-4 and not succ())

    # ================= 12-14. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.rocker_cabinet")
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

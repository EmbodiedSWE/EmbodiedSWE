"""Smoke battery for MarbleRouterScene (sim_gen task `scene_b_i305`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
physically drives / constructs a wrong outcome and asserts the rubric refuses
it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — marble waiting behind the shut gate, path fins at
                        their WRONG stops, score ~0, no success.
 2. randomization A   — target bin t varies across seeded resets and the
                        LIVE marker tile tracks the sampled bin every time.
 3. randomization B   — fin start angles and marble start jitter vary and
                        the LIVE bodies track the sample (readback).
 4. null policy       — 300 idle steps: marble still waits, gate shut,
                        score ~0, no success.
 5. seed reflex       — the CALVIN move: yank the slider (gate) open WITHOUT
                        setting the switches. PHYSICAL: the same force servo
                        the solve uses really opens the gate (asserted) and
                        the marble really rolls out (asserted) — into the
                        WRONG half; no latch fires, score ~0, no success.
                        Irreversible failure demonstrated.
 6. one fin wrong     — fin A thrown correctly (real torque servo, asserted)
                        but path fin B left wrong; gate pulled: the marble
                        takes the correct A branch, lands in the SIBLING bin;
                        fins_correct never True, no credit, no success.
 7. out of order      — gate pulled FIRST (marble commits to a wrong bin),
                        the fins thrown correctly AFTER: fins now correct and
                        gate open, but the order-aware latch chain refuses —
                        routed, released, branched all stay False, score ~0,
                        no success.
 8. wrong bin rest    — marble CONSTRUCTED at rest in the sibling bin
                        (correct half, wrong B child): settled, in a bin,
                        only the bin identity fails — no success; stays False
                        while it settles against the end wall.
 9. short of the box  — marble at rest in the TARGET lane but upstream of
                        the success box (x < x_lo): judged without stepping —
                        no success.
10. rolling through   — marble INSIDE the target box but moving 1.2 m/s
                        downhill (a transit, not a rest): the settle gate
                        refuses — judged without stepping, no success.
11. on-roof cheat     — marble at rest on the ROOF directly over the target
                        bin: the board-frame z clause refuses — judged
                        without stepping, no success.
12. ground-by-marker  — marble at rest on the GROUND next to the marker tile
                        (world-frame look-alike): board-frame x and z refuse
                        — no success, also after 20 settle steps.
13. rejection audit   — success() observed False at every step of the battery.
14. final no-NaN.
15. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.scene_b_i305.smoke --headless
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

# ----- module state wired up in main() ----------------------------------------------------------
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


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    b = scene.ball_b()[0]
    print(f"[smoke] {tag:16s} | t={int(float(scene.layout[0, 0]))} "
          f"thA={float(scene.fin_angle(scene.fin_a)[0]):+.3f} "
          f"thBL={float(scene.fin_angle(scene.fin_bl)[0]):+.3f} "
          f"thBR={float(scene.fin_angle(scene.fin_br)[0]):+.3f} "
          f"gate={float(scene.gate_y()[0]):+.4f} "
          f"ball=({float(b[0]):+.3f},{float(b[1]):+.3f},{float(b[2]):+.3f}) "
          f"fins_ok={bool(scene.fins_correct()[0])} "
          f"latch(r/g/b)=({int(scene._routed_l[0])},{int(scene._released_l[0])},"
          f"{int(scene._branched_l[0])}) "
          f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
          f"frames={len(_REC['frames'])}", flush=True)


# ----- main --------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.marble_router")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    psi_a, psi_b = c.psi_a(), c.psi_b()
    p = math.radians(c.pitch_deg)
    n_w = torch.tensor([math.sin(p), 0.0, math.cos(p)], device=device)
    zero = torch.zeros(n, 1, 3, device=device)

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((1.45, -1.15, 0.95)) + o),
                                tuple(np.array((0.40, 0.00, 0.30)) + o),
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

    def lay_t() -> int:
        return int(float(scene.layout[0, 0]))

    def lay_half() -> float:
        return float(scene.layout[0, 1])

    def target_y() -> float:
        return float(c.bin_ys()[lay_t()])

    def score0() -> bool:
        return float(scene.score()[0]) <= 0.05

    def no_latch() -> bool:
        return not (bool(scene._routed_l[0]) or bool(scene._released_l[0])
                    or bool(scene._branched_l[0]))

    def write_ball(bx: float, by: float, bz: float, vel_w=None) -> None:
        ids = _all_ids()
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0], loc[:, 1], loc[:, 2] = bx, by, bz
        st = torch.zeros(n, 13, device=device)
        st[:, :3] = scene._b2w(loc, ids)
        st[:, 3] = 1.0
        if vel_w is not None:
            st[:, 7:10] = torch.tensor(vel_w, device=device)
        scene.ball.write_root_state_to_sim(st, ids)
        _refresh()

    def fin_servo(fin, tgt: float, label: str) -> None:
        """The solve's PD torque about the fin's hinge axis (body z)."""
        kp, kd, clamp = 0.02, 0.004, 0.008
        for i in range(400):
            th = scene.fin_angle(fin)
            w = (fin.data.root_ang_vel_w * n_w).sum(dim=-1)
            tau = (kp * (torch.full_like(th, tgt) - th) - kd * w).clamp(-clamp, clamp)
            t_b = torch.zeros(n, 1, 3, device=device)
            t_b[:, 0, 2] = tau
            fin.set_external_force_and_torque(zero, t_b)
            _step(1)
            if i > 10 and abs(float(th[0]) - tgt) < 0.030 and abs(float(w[0])) < 0.4:
                break
        fin.set_external_force_and_torque(zero, zero)
        _step(30)
        print(f"[smoke] fin {label}: target {tgt:+.3f} -> "
              f"{float(scene.fin_angle(fin)[0]):+.3f}", flush=True)

    def pull_gate() -> None:
        """The solve's PD force along the gate's prismatic axis (body y)."""
        kp, kd, clamp, tgt = 30.0, 3.0, 4.0, 0.095
        for i in range(600):
            y = scene.gate_y()
            v = scene.gate.data.root_lin_vel_w[:, 1]
            fy = (kp * (torch.full_like(y, tgt) - y) - kd * v).clamp(-clamp, clamp)
            f_b = torch.zeros(n, 1, 3, device=device)
            f_b[:, 0, 1] = fy
            scene.gate.set_external_force_and_torque(f_b, zero)
            _step(1)
            if i > 20 and float(scene.ball_b()[0, 0]) > 0.30:
                break
        scene.gate.set_external_force_and_torque(zero, zero)

    # ================= 1. settle / no-NaN =========================================================
    env.reset(seed=11)
    _step(150)
    _report("settle")
    check("settle/no-NaN: marble waits behind shut gate, path fins wrong, score ~0",
          bool(scene._finite()[0]) and bool(scene.ball_waiting()[0])
          and float(scene.gate_y()[0]) < c.gate_shut_y
          and not bool(scene.fins_correct()[0]) and score0() and not succ())

    # ================= 2. randomization A: target bin + marker readback ==========================
    ts, marker_ok = [], True
    for s in range(20, 30):
        env.reset(seed=s)
        _step(3)
        _refresh()
        ts.append(lay_t())
        my_live = float(scene.marker.data.root_pos_w[0, 1] - scene.env_origins[0, 1])
        my_lay = float(scene.layout[0, 7])
        marker_ok &= abs(my_live - my_lay) < 1e-3
        marker_ok &= abs(my_lay - target_y()) <= c.marker_jit + 1e-6
    print(f"[smoke] rand A: targets {ts}", flush=True)
    check("randomization A: target bin varies (>=3 distinct) and the live marker "
          "tile tracks the sampled bin",
          len(set(ts)) >= 3 and marker_ok)

    # ================= 3. randomization B: fin starts + marble jitter readback ===================
    bxs, offs, track_ok = [], [], True
    for s in range(40, 50):
        env.reset(seed=s)
        _refresh()
        lay = scene.layout[0]
        bxs.append(float(lay[5]))
        off_fin = scene.fin_br if lay[1] > 0 else scene.fin_bl
        off_th = float(lay[4] if lay[1] > 0 else lay[3])
        offs.append(1.0 if off_th > 0 else -1.0)
        track_ok &= abs(float(scene.fin_angle(scene.fin_a)[0]) - float(lay[2])) < 0.03
        track_ok &= abs(float(scene.fin_angle(off_fin)[0]) - off_th) < 0.03
        b = scene.ball_b()[0]
        track_ok &= abs(float(b[0]) - float(lay[5])) < 0.02
        track_ok &= abs(float(b[1]) - float(lay[6])) < 0.02
    print(f"[smoke] rand B: ball_x std={np.std(bxs):.4f} off-fin signs {offs}",
          flush=True)
    check("randomization B: marble start varies (std > 8 mm), off-path fin takes "
          "both stops, live fins/marble track the sample",
          float(np.std(bxs)) > 0.008 and len(set(offs)) == 2 and track_ok)

    # ================= 4. null policy =============================================================
    env.reset(seed=77)
    _step(300)
    _report("null")
    check("null policy: 300 idle steps -> marble still waiting, gate shut, score ~0",
          bool(scene.ball_waiting()[0]) and float(scene.gate_y()[0]) < c.gate_shut_y
          and score0() and no_latch() and not succ())

    # ================= 5. seed reflex: yank the gate, fins unset ==================================
    env.reset(seed=101)
    _step(120)
    pull_gate()
    gate_open = float(scene.gate_y()[0]) > c.gate_open_y
    _step(450)
    _report("gate-only")
    b = scene.ball_b()[0]
    check("seed reflex: gate really opened, marble really rolled out — into the "
          "WRONG half; no latch, score ~0, no success, irreversibly failed",
          gate_open and float(b[0]) > c.bins_x0
          and float(b[1]) * lay_half() < 0
          and bool(scene.settled()[0]) and no_latch() and score0() and not succ())

    # ================= 6. one fin wrong ===========================================================
    env.reset(seed=131)
    _step(120)
    fin_servo(scene.fin_a, lay_half() * psi_a, "A (only)")
    a_ok = abs(float(scene.fin_angle(scene.fin_a)[0]) - lay_half() * psi_a) < c.fin_tol
    fins_still_wrong = not bool(scene.fins_correct()[0])
    pull_gate()
    _step(450)
    _report("one-fin")
    b = scene.ball_b()[0]
    check("one fin wrong: fin A really thrown (asserted), B wrong -> marble takes "
          "the correct half but the SIBLING bin; no credit, no success",
          a_ok and fins_still_wrong and float(b[0]) > c.bins_x0
          and float(b[1]) * lay_half() > 0
          and abs(float(b[1]) - target_y()) > c.y_tol
          and bool(scene.settled()[0]) and no_latch() and score0() and not succ())

    # ================= 7. out of order ============================================================
    env.reset(seed=163)
    _step(120)
    t7, half7 = lay_t(), lay_half()
    sgn_b7 = 1.0 if t7 in (0, 2) else -1.0
    pull_gate()
    _step(450)
    fin_servo(scene.fin_a, half7 * psi_a, "A (late)")
    fin_servo(scene.fin_bl if half7 > 0 else scene.fin_br, sgn_b7 * psi_b, "B (late)")
    _step(60)
    _report("out-of-order")
    check("out of order: fins thrown correctly AFTER the release — fins_correct "
          "holds and the gate is open, but the order-aware chain refuses all "
          "three latches; score ~0, no success",
          bool(scene.fins_correct()[0]) and float(scene.gate_y()[0]) > c.gate_open_y
          and no_latch() and score0() and not succ())

    # ================= 8. wrong-bin rest (construct) ==============================================
    env.reset(seed=197)
    _step(60)
    sib = {0: 1, 1: 0, 2: 3, 3: 2}[lay_t()]
    y_sib = float(c.bin_ys()[sib])
    write_ball(0.945, y_sib, c.ball_r + 0.001)
    _report("sibling-bin")
    in_bin_but_wrong = (not succ()) and bool(scene.settled()[0])
    _step(40)
    _report("sibling-rest")
    b = scene.ball_b()[0]
    check("wrong bin: marble at rest in the SIBLING bin (in a bin box in x/z, "
          "only the bin identity fails) -> no success, also after settling",
          in_bin_but_wrong and float(b[0]) > c.x_lo and float(b[0]) < c.x_hi
          and abs(float(b[1]) - y_sib) < c.y_tol
          and abs(float(b[1]) - target_y()) > c.y_tol
          and bool(scene.settled()[0]) and not succ())

    # ================= 9. short of the box (judged without stepping) ==============================
    write_ball(0.840, target_y(), c.ball_r + 0.001)
    short_no = not succ()
    short_settled = bool(scene.settled()[0])
    check("short of the box: marble at rest in the TARGET lane at x=0.84 < x_lo "
          "-> the x clause refuses (judged without stepping)",
          short_no and short_settled and not bool(scene.in_target_bin()[0]))

    # ================= 10. rolling through (judged without stepping) ==============================
    cp, sp = math.cos(p), math.sin(p)
    write_ball(0.900, target_y(), c.ball_r + 0.001, vel_w=(1.2 * cp, 0.0, -1.2 * sp))
    check("rolling through: marble INSIDE the target box at 1.2 m/s -> the "
          "settle gate refuses (judged without stepping)",
          bool(scene.in_target_bin()[0]) and not bool(scene.settled()[0])
          and not succ())

    # ================= 11. on-roof cheat (judged without stepping) ================================
    write_ball(0.920, target_y(), c.roof_z1 + c.ball_r + 0.001)
    check("on-roof cheat: marble at rest on the ROOF over the target bin -> the "
          "board-frame z clause refuses (judged without stepping)",
          not bool(scene.in_target_bin()[0]) and bool(scene.settled()[0])
          and not succ())

    # ================= 12. ground-by-marker look-alike ============================================
    ids = _all_ids()
    st = torch.zeros(n, 13, device=device)
    st[:, 0] = scene.env_origins[:, 0] + c.marker_x
    st[:, 1] = scene.env_origins[:, 1] + scene.layout[:, 7]
    st[:, 2] = scene.env_origins[:, 2] + c.ball_r + 0.001
    st[:, 3] = 1.0
    scene.ball.write_root_state_to_sim(st, ids)
    _refresh()
    ground_no = not succ()
    _step(20)
    _report("by-marker")
    check("ground-by-marker: marble resting ON the marker tile outside the yard "
          "-> board-frame x/z refuse, also after 20 settle steps",
          ground_no and not bool(scene.in_target_bin()[0]) and not succ())

    # ================= 13-15. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    env.reset(seed=211)
    _step(60)
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.marble_router")
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
        traceback = __import__("traceback")
        traceback.print_exc()
        print(f"SIM_GEN_SMOKE: FAIL ({type(e).__name__}: {e})", flush=True)
        os._exit(1)

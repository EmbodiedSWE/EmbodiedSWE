"""Smoke battery for CrankFerryScene (sim_gen task `native_libero_i365`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS (or physically drives) a wrong outcome and asserts the rubric
refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — cage resting at its sampled cage_x, cubes on the apron,
                        score ~0, no success.
 2. randomization A   — the sampled crank angle theta0 (via cage_x) varies
                        across resets and the settled cage tracks it (readback).
 3. randomization B   — cargo and decoy xy poses vary across resets (readback).
 4. null policy       — 240 idle steps -> the friction-held mechanism stays
                        put, cubes stay on the apron, score ~0, no success.
 5. SEED strategy     — the seed's whole skill (pick up the cube, carry it over
                        the goal, drop it) executed as far as physics allows:
                        cargo teleported right above the vault and RELEASED for
                        a real gravity drop. It lands straddling the 32 mm roof
                        slit (asserted: it really fell) and can never enter the
                        sealed vault; no drop latch, score ~0, no success.
 6. wrong object      — the blue DECOY constructed inside the vault (settled on
                        the vault floor, asserted): identity matters — score ~0,
                        no success.
 7. out-of-order      — cube pushed through the window EARLY (constructed loose
                        in the lane, yoke at the hole end), then the crank driven
                        for REAL back to the window: the returning cage PLOWS the
                        loose cube into the near-end pocket (asserted: it really
                        got shoved past the cage face). Never aboard, at most
                        align credit, no success — order matters.
 8. near miss         — cargo constructed at rest in the lane just short of the
                        hole edge (fully supported): no drop latch, no success.
 9. in-vault, moving  — cargo written inside the vault box WITH velocity: the
                        settle gate refuses success on the live moving state
                        (judged without stepping; then retracted).
10. latched credit    — the REAL solve prefix: crank to the dead center (real
                        torque), load the cargo through the window (real push),
                        then the cargo STOLEN back to the apron: align+loaded
                        latches survive, success does not.
11. empty ferry       — the crank spun >5 rad for real, the empty cage sweeping
                        (asserted >25 cm of travel): at most align credit, no
                        success — cranking without loading is worthless.
12. rejection audit   — success() observed False at every step of the battery.
13. final no-NaN.
14. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.native_libero_i365.smoke --headless
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


def _write_body(body, pos_w: torch.Tensor, yaw: float = 0.0, vel_x: float = 0.0) -> None:
    st = torch.zeros(_ENV.num_envs, 13, device=_ENV.device)
    st[:, 0:3] = pos_w
    st[:, 3] = math.cos(yaw / 2)
    st[:, 6] = math.sin(yaw / 2)
    st[:, 7] = vel_x
    body.write_root_state_to_sim(st, _all_ids())
    _refresh()


def _report(tag: str) -> None:
    scene = _ENV.scene
    _refresh()
    p = (scene.cargo.data.root_pos_w - scene.env_origins)[0]
    print(f"[smoke] {tag:16s} | cage_x={float(scene.cage_x()[0]):+.4f} "
          f"x0={float(scene.x0[0]):+.4f} "
          f"cargo=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
          f"latch(a/l/d)=({int(scene._align[0])},{int(scene._loaded[0])},"
          f"{int(scene._dropped[0])}) ferry={float(scene._ferry[0]):.2f} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.crank_ferry")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    dt = 1.0 / 120.0
    from isaaclab.utils.math import quat_apply_inverse

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.55, 1.05, 0.85)) + o),
                                tuple(np.array((0.20, 0.00, 0.20)) + o),
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

    def write_yoke(theta: float) -> None:
        """Teleport the WHOLE linkage consistently (one write): disc at yaw
        `theta`, cage at the implied cage_x = crank_x + crank_r*cos(theta)."""
        _write_body(scene.disc, origin(c.crank_x, 0.0, c.crank_z), yaw=theta)
        _write_body(scene.cage, origin(c.crank_x + c.crank_r * math.cos(theta), 0.0, 0.0))
        _step(30)

    def crank(dir_: float, *, done=None, steps: int, w_des: float = 0.8,
              taumax: float = 1.5, label: str = "") -> float:
        """REAL actuation: a velocity-servo torque about the crank axle (what a
        hand pushing the red handle peg around its circle applies). Body-frame
        (0,0,tau) is valid: the disc is yaw-only. Returns |rotation| in rad."""
        rot = 0.0
        for _ in range(steps):
            w = float(scene.disc.data.root_ang_vel_w[0, 2])
            rot += abs(w) * dt
            tau = max(-taumax, min(taumax, 0.35 * (dir_ * w_des - w)))
            t = torch.zeros(n, 1, 3, device=device)
            t[:, 0, 2] = tau
            scene.disc.set_external_force_and_torque(zero, t)
            _step(1)
            if done is not None and done():
                break
        for _ in range(180):  # active brake: servo the crank to rest
            w = float(scene.disc.data.root_ang_vel_w[0, 2])
            if abs(w) < 0.04:
                break
            tau = max(-1.2, min(1.2, 0.6 * (0.0 - w)))
            t = torch.zeros(n, 1, 3, device=device)
            t[:, 0, 2] = tau
            scene.disc.set_external_force_and_torque(zero, t)
            _step(1)
        scene.disc.set_external_force_and_torque(zero, zero)
        _step(30)
        print(f"[smoke] crank {label}: cage_x={float(scene.cage_x()[0]):+.4f} "
              f"rot={rot:.2f} rad", flush=True)
        return rot

    def push_cargo(steps: int = 480) -> None:
        """REAL loading push: a gentle velocity-servo force on the cargo (what a
        fingertip sliding the cube across the apron applies) — -y through the
        window, x-centering onto the cage. Force-limited to ~2 N."""
        for _ in range(steps):
            p = (scene.cargo.data.root_pos_w - scene.env_origins)[0]
            if float(p[1]) < -0.005:
                break
            v = scene.cargo.data.root_lin_vel_w
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 0] = (1.5 * (scene.cage_x() - (scene.cargo.data.root_pos_w
                         - scene.env_origins)[:, 0]) - 0.6 * v[:, 0]).clamp(-0.6, 0.6)
            # sized to beat ~0.4 N apron friction at rest; K*dt/m = 0.49 (stable)
            f_w[:, 1] = (7.0 * (-0.15 - v[:, 1])).clamp(-2.0, 2.0)
            f_b = quat_apply_inverse(scene.cargo.data.root_quat_w, f_w)
            scene.cargo.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
        scene.cargo.set_external_force_and_torque(zero, zero)
        _step(60)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    p1 = (scene.cargo.data.root_pos_w - scene.env_origins)[0]
    check("settle/no-NaN: cage resting at its sampled cage_x, cargo on the "
          "apron, score ~0, no success",
          bool(scene._finite()[0])
          and abs(float(scene.cage_x()[0]) - float(scene.x0[0])) < 0.010
          and float(p1[1]) > 0.06 and 0.14 < float(p1[2]) < 0.18
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    x0s, cages, xys = [], [], []
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        x0s.append(float(scene.x0[0]))
        cages.append(float(scene.cage_x()[0]))
        cw = (scene.cargo.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist()
        dw = (scene.decoy.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist()
        xys.append(cw + dw)
    x0_span = max(x0s) - min(x0s)
    tracks = all(abs(cx - x0) < 0.010 for cx, x0 in zip(cages, x0s))
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: x0={[f'{d:.4f}' for d in x0s]} span={x0_span:.4f} "
          f"tracks={tracks} cube_xystd={xystd:.4f}", flush=True)
    check("randomization A: the sampled crank angle (via cage_x) varies across "
          "resets and the settled cage tracks it every time",
          x0_span > 0.030 and tracks)
    check("randomization B: cargo and decoy xy poses vary across resets",
          xystd > 0.008)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    p4 = (scene.cargo.data.root_pos_w - scene.env_origins)[0]
    check("null policy: 240 idle steps -> the friction-held mechanism stays "
          "put, cargo stays on the apron, score ~0, no success",
          abs(float(scene.cage_x()[0]) - float(scene.x0[0])) < 0.010
          and float(p4[1]) > 0.06
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (carry over the goal and drop, for real) ==================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    # the "carry" is the teleport (transport); the DROP is real physics
    _write_body(scene.cargo, origin(0.5 * (c.hole_x0 + c.hole_x1), 0.0, 0.330))
    _step(240)
    _report("seed-drop")
    p5 = (scene.cargo.data.root_pos_w - scene.env_origins)[0]
    check("SEED strategy: cargo released right above the vault really falls "
          "and lands straddling the 32 mm roof slit — it can never enter the "
          "sealed vault; no drop latch, score ~0, no success",
          0.20 <= float(p5[2]) <= 0.30
          and abs(float(p5[0]) - 0.5 * (c.hole_x0 + c.hole_x1)) < 0.06
          and not bool(scene._dropped[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. wrong object (decoy constructed in the vault) ===========================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    _write_body(scene.decoy, origin(0.385, 0.0, c.decoy_s / 2 + 0.002))
    _step(120)
    _report("wrong-object")
    d6 = (scene.decoy.data.root_pos_w - scene.env_origins)[0]
    check("wrong object: the blue decoy settled on the vault floor (it is "
          "really in the vault) — identity matters: score ~0, no success",
          float(d6[2]) < 0.07 and abs(float(d6[1])) < c.vault_gy
          and c.vault_gx0 <= float(d6[0]) <= c.vault_gx1
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 7. out-of-order (early load -> plowed into the pocket) =====================
    torch.manual_seed(61)
    env.reset()
    _step(60)
    write_yoke(0.0)  # cage parked at the hole end (theta=0)
    _write_body(scene.cargo, origin(0.5 * (c.win_x0 + c.win_x1), 0.0,
                                    c.floor_z1 + c.cargo_s / 2 + 0.003))
    _step(30)
    cargo_before = float((scene.cargo.data.root_pos_w - scene.env_origins)[0, 0])
    crank(+1.0, done=lambda: float(scene.cage_x()[0]) <= scene.XL + 0.005,
          steps=1100, label="return-to-window")
    _report("out-of-order")
    p7 = (scene.cargo.data.root_pos_w - scene.env_origins)[0]
    check("out-of-order: a cube pushed through the window EARLY lies loose in "
          "the lane; the crank driven for real back to the window PLOWS it "
          "into the near-end pocket (it really got shoved) — never aboard, at "
          "most align credit, no success",
          float(scene.cage_x()[0]) <= scene.XL + 0.010
          and cargo_before - float(p7[0]) >= 0.030
          and float(p7[0]) <= 0.048 and float(p7[2]) < 0.20
          and not bool(scene._loaded[0])
          and float(scene.score()[0]) <= c.w_align + 1e-4 and not succ())

    # ================= 8. near miss (resting just short of the hole edge) =========================
    torch.manual_seed(71)
    env.reset()
    _step(60)
    write_yoke(math.pi)  # cage at the window, out of the way
    _write_body(scene.cargo, origin(0.315, 0.0, c.floor_z1 + c.cargo_s / 2 + 0.003))
    _step(120)
    _report("near-miss")
    p8 = (scene.cargo.data.root_pos_w - scene.env_origins)[0]
    check("near miss: cargo at rest in the lane just short of the hole edge "
          "(fully supported) — no drop latch, no success",
          0.29 <= float(p8[0]) <= 0.335 and 0.14 < float(p8[2]) < 0.18
          and not bool(scene._dropped[0])
          and float(scene.score()[0]) <= c.w_align + 1e-4 and not succ())

    # ================= 9. in-vault but moving (settle gate) =======================================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    _write_body(scene.cargo, origin(0.385, 0.0, c.cargo_s / 2 + 0.002), vel_x=0.35)
    p9 = (scene.cargo.data.root_pos_w - scene.env_origins)[0]
    in_box = (c.vault_gx0 <= float(p9[0]) <= c.vault_gx1
              and abs(float(p9[1])) <= c.vault_gy and float(p9[2]) <= c.vault_gz)
    moving_rejected = not succ()  # judged WITHOUT stepping: in the box but moving
    _write_body(scene.cargo, origin(0.05, 0.15, 0.156))  # retract before stepping
    _step(90)
    _report("in-vault-moving")
    check("in-vault but moving: cargo written inside the vault box with "
          "velocity — the settle gate refuses success on the moving state",
          in_box and moving_rejected
          and float(scene.score()[0]) <= c.w_drop + 1e-4 and not succ())

    # ================= 10. latched credit (real align + real load, then stolen) ===================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    dir10 = 1.0 if float(scene.th0[0]) >= 0.0 else -1.0
    crank(dir10, done=lambda: abs(float(scene.cage_x()[0]) - scene.XL) <= 0.005,
          steps=900, label="to-window")
    aligned = bool(scene._align[0])
    _write_body(scene.cargo, origin(0.5 * (c.win_x0 + c.win_x1), 0.088,
                                    c.apron_z1 + c.cargo_s / 2 + 0.003))
    _step(30)
    push_cargo(480)
    loaded = bool(scene._loaded[0])
    held = abs(float(scene.cage_x()[0]) - scene.XL) <= c.align_tol + 0.004
    _write_body(scene.cargo, origin(0.05, 0.15, 0.156))  # steal the cargo back
    _step(90)
    _report("latched-credit")
    s10 = float(scene.score()[0])
    check("latched credit: the real solve prefix (crank to dead center, push "
          "the cargo aboard — the dead center held) then the cargo stolen "
          "back to the apron: align+loaded latches survive, success does not",
          aligned and loaded and held
          and c.w_align + c.w_loaded - 1e-4 <= s10 <= 0.42 and not succ())

    # ================= 11. empty ferry (cranking without loading is worthless) ====================
    torch.manual_seed(101)
    env.reset()
    _step(60)
    lo, hi = float(scene.cage_x()[0]), float(scene.cage_x()[0])

    def track_span() -> bool:
        nonlocal lo, hi
        cx = float(scene.cage_x()[0])
        lo, hi = min(lo, cx), max(hi, cx)
        return False  # never stop early: spin the full budget

    rot11 = crank(+1.0, done=track_span, steps=900, label="empty-ferry")
    _report("empty-ferry")
    p11 = (scene.cargo.data.root_pos_w - scene.env_origins)[0]
    check("empty ferry: the crank spun for real (>5 rad, the empty cage swept "
          ">25 cm) — at most align credit, cubes untouched on the apron, no "
          "success",
          rot11 >= 5.0 and hi - lo >= 0.25 and float(p11[1]) > 0.06
          and float(scene.score()[0]) <= c.w_align + 1e-4 and not succ())

    # ================= 12-14. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.crank_ferry")
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

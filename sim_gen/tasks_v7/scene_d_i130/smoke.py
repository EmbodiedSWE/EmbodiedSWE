"""Smoke battery for CarouselVaultScene (sim_gen task `scene_d_i130`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS (or physically drives) a settled wrong outcome and asserts the
rubric refuses it. success() must never fire anywhere in the battery.

 1. settle/no-NaN     — shutter closed at 0, disc at its sampled yaw, prize
                        riding its pocket, score ~0, no success.
 2. randomization A   — the disc's sampled start yaw varies across resets
                        (>1.5 rad span) and BOTH the live disc yaw and the
                        prize's pocket seat track the sample every time.
 3. randomization B   — the pad's sampled ground xy varies across resets and
                        the live pad body tracks the sample (readback).
 4. null policy       — 240 idle steps -> vault sealed, cube captive, score
                        ~0, no success.
 5. closed-shutter cap— disc CONSTRUCTED indexed (whole assembly written
                        together) but shutter left CLOSED; the same REAL lift
                        force as solve.py pushes the cube up: it rises off its
                        seat (asserted — the force pathway is live) and jams
                        on the shutter plate far below the out plane; the out
                        latch never fires, no success.
 6. un-indexed cap    — shutter WRITTEN open (consistent with its prismatic)
                        but the rotor assembly CONSTRUCTED at yaw pi (red
                        pocket fully under solid roof); the REAL lift force
                        pushes the cube up: it rises (asserted) and jams on
                        the ROOF, far below the out plane; no out latch, no
                        success.
 7. seed reflex       — shutter-only: the REAL solve-style push slides the
                        shutter fully open and it STAYS open hands-off, but
                        that is the whole CALVIN slider move — score exactly
                        the open credit (0.15), no success.
 8. wrong object      — the BLUE decoy constructed resting on the pad: the
                        rubric keys the red cube — score ~0, no success.
 9. near miss         — the red cube constructed settled on the GROUND 11 cm
                        from the pad centre: out credit only (capped), on_pad
                        False, no success.
10. on-pad but moving — the red cube written AT the pad rest pose WITH lateral
                        velocity: the settle gate refuses success on the
                        moving state (judged without stepping; then removed).
11. hover above pad   — the red cube written 5 cm ABOVE the pad rest height:
                        the z gate refuses success (judged without stepping;
                        then removed).
12. rejection audit   — success() observed False at every step of the battery.
13. final no-NaN.
14. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.scene_d_i130.smoke --headless
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

# ----- servo gains (the same limited hand as solve.py) ------------------------------------------
KP_S, KD_S, F_S = 60.0, 8.0, 8.0        # shutter knob push
KP_B, KD_B, F_B = 8.0, 0.8, 2.5         # cube lift

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
    p = (scene.prize.data.root_pos_w - scene.env_origins)[0]
    print(f"[smoke] {tag:16s} | open={float(scene.shutter_open_amt()[0]):+.4f} "
          f"yaw={float(scene.disc_yaw()[0]):+.3f} "
          f"prize=({float(p[0]):+.3f},{float(p[1]):+.3f},{float(p[2]):+.3f}) "
          f"latch(o/i/x)=({int(scene._open_l[0])},{int(scene._index_l[0])},"
          f"{int(scene._out_l[0])}) settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main --------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.carousel_vault")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.05, -0.85, 0.75)) + o),
                                tuple(np.array((0.05, -0.05, 0.12)) + o),
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

    zero = torch.zeros(n, 1, 3, device=device)

    def write_body(body, dx: float, dy: float, dz: float, yaw: float = 0.0,
                   vel_x: float = 0.0) -> None:
        st = torch.zeros(n, 13, device=device)
        st[:, 0] = scene.env_origins[:, 0] + dx
        st[:, 1] = scene.env_origins[:, 1] + dy
        st[:, 2] = scene.env_origins[:, 2] + dz
        st[:, 3] = math.cos(yaw / 2.0)
        st[:, 6] = math.sin(yaw / 2.0)
        st[:, 7] = vel_x
        body.write_root_state_to_sim(st, _all_ids())
        _refresh()

    def write_assembly(yaw: float) -> None:
        """Construct the whole rotor assembly (disc + both pocketed cubes) at
        `yaw` TOGETHER — teleporting one body of a linkage gets depenetrated
        back by the rest (memory: teleport the whole linkage)."""
        bz = c.disc_z + c.disc_t / 2 + c.block_s / 2 + 0.002
        write_body(scene.disc, 0.0, 0.0, c.disc_z, yaw=yaw)
        write_body(scene.prize, c.r_p * math.cos(yaw), c.r_p * math.sin(yaw), bz, yaw=yaw)
        write_body(scene.decoy, -c.r_p * math.cos(yaw), -c.r_p * math.sin(yaw), bz, yaw=yaw)

    def push_shutter(y_to: float, steps: int, tail: int) -> None:
        """REAL actuation: the same limited PD y-force on the shutter as solve.py."""
        y0 = float(scene.shutter_open_amt()[0])
        for i in range(steps + tail):
            a = min(1.0, i / max(steps, 1))
            y_t = y0 + (y_to - y0) * a
            y = scene.shutter_open_amt()
            vy = scene.shutter.data.root_lin_vel_w[:, 1]
            f_w = torch.zeros(n, 3, device=device)
            f_w[:, 1] = (KP_S * (y_t - y) - KD_S * vy).clamp(min=-F_S, max=F_S)
            f_b = quat_apply_inverse(scene.shutter.data.root_quat_w, f_w)
            scene.shutter.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
            if a >= 1.0 and float(scene.shutter_open_amt()[0]) >= c.open_thresh + 0.005 \
                    and abs(float(vy[0])) < 0.05:
                break
        scene.shutter.set_external_force_and_torque(zero, zero)
        _step(60)

    def lift_probe(z_to: float = 0.30, steps: int = 240, tail: int = 120) -> tuple[float, float]:
        """REAL actuation: the same limited PD lift (+ gravity ff) on the red
        cube as solve.py; xy held at the capture point. Returns (rise, z_max)."""
        p0 = (scene.prize.data.root_pos_w - scene.env_origins).clone()
        z_start = float(p0[0, 2])
        z_max = z_start
        for i in range(steps + tail):
            a = min(1.0, i / max(steps, 1))
            p_t = p0.clone()
            p_t[:, 2] = p0[:, 2] + (z_to - p0[:, 2]) * a
            p = scene.prize.data.root_pos_w - scene.env_origins
            v = scene.prize.data.root_lin_vel_w
            f_w = KP_B * (p_t - p) - KD_B * v
            f_w[:, 2] += c.block_mass * 9.81
            f_w = f_w.clamp(min=-F_B, max=F_B)
            f_b = quat_apply_inverse(scene.prize.data.root_quat_w, f_w)
            scene.prize.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
            z_max = max(z_max, float((scene.prize.data.root_pos_w - scene.env_origins)[0, 2]))
        scene.prize.set_external_force_and_torque(zero, zero)
        _step(60)
        return z_max - z_start, z_max

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    th0 = float(scene.layout[0, 0])
    th0_w = math.atan2(math.sin(th0), math.cos(th0))
    yaw1 = float(scene.disc_yaw()[0])
    pz1 = (scene.prize.data.root_pos_w - scene.env_origins)[0]
    pk1 = scene.red_pocket_xy()[0]
    check("settle/no-NaN: shutter closed at 0, disc at its sampled yaw, prize "
          "riding its pocket, score ~0, no success",
          bool(scene._finite()[0])
          and float(scene.shutter_open_amt()[0]) < 0.01
          and abs(math.atan2(math.sin(yaw1 - th0_w), math.cos(yaw1 - th0_w))) < 0.06
          and float((pz1[:2] - pk1).norm()) < 0.02
          and not bool(scene.indexed()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    thetas, tracks, pads, pad_tracks = [], [], [], []
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        t = float(scene.layout[0, 0])
        tw = math.atan2(math.sin(t), math.cos(t))
        yw = float(scene.disc_yaw()[0])
        thetas.append(t)
        pz = (scene.prize.data.root_pos_w - scene.env_origins)[0]
        pk = scene.red_pocket_xy()[0]
        tracks.append(abs(math.atan2(math.sin(yw - tw), math.cos(yw - tw))) < 0.06
                      and float((pz[:2] - pk).norm()) < 0.02)
        pd = (scene.pad.data.root_pos_w - scene.env_origins)[0]
        pads.append([float(scene.layout[0, 1]), float(scene.layout[0, 2])])
        pad_tracks.append(abs(float(pd[0]) - pads[-1][0]) < 0.005
                          and abs(float(pd[1]) - pads[-1][1]) < 0.005)
    th_span = max(thetas) - min(thetas)
    pad_std = float(np.std(np.asarray(pads), axis=0).mean())
    print(f"[smoke] readback: theta0 span={th_span:.2f} rad tracks={all(tracks)} "
          f"pad_std={pad_std:.3f} pad_tracks={all(pad_tracks)}", flush=True)
    check("randomization A: the disc's sampled start yaw varies across resets "
          "(>1.5 rad span) and the live disc + pocketed prize track it every time",
          th_span > 1.5 and all(tracks))
    check("randomization B: the pad's sampled ground xy varies across resets "
          "and the live pad body tracks the sample",
          pad_std > 0.008 and all(pad_tracks))

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    pz4 = (scene.prize.data.root_pos_w - scene.env_origins)[0]
    check("null policy: 240 idle steps -> shutter closed, cube captive in its "
          "pocket, score ~0, no success",
          float(scene.shutter_open_amt()[0]) < 0.01
          and float(pz4[2]) < c.roof_z0
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. closed-shutter cap (real lift vs the shutter plate) ====================
    torch.manual_seed(41)
    env.reset()
    _step(120)
    write_assembly(0.0)  # constructed indexed — but the shutter stays CLOSED
    _step(60)
    assert bool(scene.indexed()[0]), "probe setup: constructed assembly must be indexed"
    rise5, zmax5 = lift_probe()
    _report("closed-cap")
    check("closed-shutter cap: with the pocket indexed but the shutter CLOSED, "
          "the real solve-grade lift raises the cube off its seat (the force "
          "is live) yet it jams on the shutter plate far below the out plane; "
          "out never fires, no success",
          rise5 >= 0.004 and zmax5 < 0.20 and zmax5 < c.out_z - 0.02
          and not bool(scene._out_l[0])
          and float(scene.score()[0]) <= c.w_index + 1e-4 and not succ())

    # ================= 6. un-indexed cap (real lift vs the roof) =================================
    torch.manual_seed(51)
    env.reset()
    _step(120)
    write_assembly(math.pi)  # red pocket at the BACK, fully under solid roof
    write_body(scene.shutter, c.shut_cx, 0.162, c.shut_cz)  # window open
    _step(60)
    assert not bool(scene.indexed()[0]), "probe setup: disc must be un-indexed"
    rise6, zmax6 = lift_probe()
    _report("roof-cap")
    check("un-indexed cap: with the window OPEN but the red pocket at the "
          "back, the real lift raises the cube (the force is live) yet it "
          "jams on the roof far below the out plane; out never fires, no success",
          rise6 >= 0.003 and zmax6 < 0.175
          and not bool(scene._out_l[0])
          and float(scene.score()[0]) <= c.w_open + 1e-4 and not succ())

    # ================= 7. seed reflex (shutter-only = the CALVIN slider push) ====================
    torch.manual_seed(61)
    env.reset()
    _step(120)
    push_shutter(0.162, 300, 300)
    _step(120)  # hands off: the shutter must STAND open
    _report("shutter-only")
    check("seed reflex: the real push slides the shutter fully open and it "
          "stays open hands-off — but that is the whole slider move: score "
          "exactly the open credit, no success",
          bool(scene.opened()[0])
          and c.w_open - 1e-4 <= float(scene.score()[0]) <= c.w_open + 1e-4
          and not succ())

    # ================= 8. wrong object (decoy on the pad) =========================================
    torch.manual_seed(71)
    env.reset()
    _step(120)
    px, py = float(scene.layout[0, 1]), float(scene.layout[0, 2])
    write_body(scene.decoy, px, py, c.pad_h + c.block_s / 2 + 0.002)
    _step(120)
    _report("decoy-on-pad")
    check("wrong object: the BLUE decoy constructed resting on the pad — the "
          "rubric keys the red cube: score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 9. near miss (on the ground, 11 cm off the pad) ============================
    torch.manual_seed(81)
    env.reset()
    _step(120)
    px, py = float(scene.layout[0, 1]), float(scene.layout[0, 2])
    write_body(scene.prize, px + 0.11, py, c.block_s / 2 + 0.002)
    _step(120)
    _report("near-miss")
    check("near miss: the red cube settled on the GROUND 11 cm from the pad "
          "centre — out credit only (capped), on_pad False, no success",
          not bool(scene.on_pad()[0])
          and c.w_out - 1e-4 <= float(scene.score()[0]) <= 0.60 + 1e-4
          and not succ())

    # ================= 10. on-pad but moving (settle gate, judged pre-step) ======================
    torch.manual_seed(91)
    env.reset()
    _step(120)
    px, py = float(scene.layout[0, 1]), float(scene.layout[0, 2])
    write_body(scene.prize, px, py, c.pad_h + c.block_s / 2, vel_x=0.30)
    on_pad_now = bool(scene.on_pad()[0])
    moving_rejected = not succ()  # judged WITHOUT stepping: at the pad but moving
    write_body(scene.prize, 0.65, 0.45, c.block_s / 2 + 0.002)  # remove before stepping
    _step(90)
    _report("pad-moving")
    check("on-pad but moving: the red cube written AT the pad rest pose with "
          "lateral velocity — the settle gate refuses success on the moving state",
          on_pad_now and moving_rejected and not succ())

    # ================= 11. hover above pad (z gate, judged pre-step) =============================
    torch.manual_seed(101)
    env.reset()
    _step(120)
    px, py = float(scene.layout[0, 1]), float(scene.layout[0, 2])
    write_body(scene.prize, px, py, c.pad_h + c.block_s / 2 + 0.05)
    hover_rejected = not succ()  # judged WITHOUT stepping: above the pad, not ON it
    hover_off_pad = not bool(scene.on_pad()[0])
    write_body(scene.prize, 0.65, 0.45, c.block_s / 2 + 0.002)  # remove before stepping
    _step(90)
    _report("pad-hover")
    check("hover above pad: the red cube written 5 cm above the pad rest "
          "height — the z gate refuses success on the airborne state",
          hover_rejected and hover_off_pad and not succ())

    # ================= 12-14. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.carousel_vault")
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

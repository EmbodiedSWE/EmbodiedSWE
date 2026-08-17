"""Smoke battery for RamChuteCabinetScene (sim_gen task
`libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i276`) — REJECTION-ONLY:
every check either verifies basic health/randomization or CONSTRUCTS a settled
wrong outcome and asserts the rubric refuses it. success() must never fire
anywhere in the battery.

 1. settle/no-NaN     — drawer resting at its sampled opening q0, ball in its dock
                        tray, score ~0, no success.
 2. randomization A   — station yaw spans > 90 deg and xy jitters across resets.
 3. randomization B   — the drawer opening q0 varies across resets and the settled
                        drawer tracks it; the ball's dock position jitters and the
                        ball rests IN the tray every time (readback).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (push the drawer shut) executed by
                        an ORACLE force (a hand cannot even reach: the guard
                        tunnel denies access — asserted geometrically in the
                        scene): a 6 N PD push seats the drawer for real. No ram
                        was ever delivered: no success, drawer credit only (~0.55).
 6. dead ram          — the ball LAID at rest against the open drawer's face
                        (transport without the descent): it has no kinetic energy
                        and the drawer does not close; no success, delivery credit
                        only (~0.30).
 7. park near miss    — drawer constructed seated + ball at rest on the runway
                        20 mm outside the park gap tolerance: parked-clause
                        refuses; base score capped below 1 (<= 0.85 float32),
                        no success.
 8. channel guard     — drawer STOLEN out of the cabinet onto the ground, ball
                        resting in the tunnel: the drawer clause and the
                        drawer-progress latch refuse a drawer that is not in its
                        channel; delivery credit only, no success.
 9. latch survival    — ball delivered (rested against the open drawer, latch
                        fires) then RETURNED to its dock: the delivery latch
                        survives, live delivered/parked read False, no success.
10. rejection audit   — success() observed False at every step of the battery.
11. final no-NaN.
12. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i276.smoke --headless
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
    bl = scene._station_local(scene.ball.data.root_pos_w)[0]
    print(f"[smoke] {tag:18s} | q={float(scene.drawer_q()[0]):+.4f} "
          f"ball_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
          f"delivered={bool(scene.ball_delivered_now()[0])} "
          f"parked={bool(scene.ball_parked()[0])} "
          f"drawer_closed={bool(scene.drawer_closed()[0])} "
          f"in_channel={bool(scene.drawer_in_channel()[0])} "
          f"latch(de/dr)=({float(scene._fdeliver[0]):.2f},{float(scene._fdrawer[0]):.2f}) "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ram_chute_cabinet")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-1.25, -1.35, 1.05)) + o),
                                tuple(np.array((0.28, 0.08, 0.35)) + o),
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

    def station_yaw() -> torch.Tensor:
        return task_scene._yaw_of(scene.station.data.root_quat_w)

    def station_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.station.data.root_pos_w + quat_apply(
            scene.station.data.root_quat_w, loc)

    def ground(dx: float, dy: float, h: float) -> torch.Tensor:
        return (scene.env_origins[0:1]
                + torch.tensor([dx, dy, h], device=device)).expand(n, 3)

    def put_drawer(q: float) -> None:
        """CONSTRUCT: the drawer seated in its channel at opening q."""
        _write_body(scene.drawer,
                    station_world([q - c.drawer_l / 2, 0.0, c.slab_t + 0.002]),
                    task_scene._qz(station_yaw()))

    def put_ball(x: float, y: float = 0.0, z: float | None = None) -> None:
        """CONSTRUCT: the ball at rest at a station-local point (default: resting
        on the runway floor)."""
        _write_body(scene.ball,
                    station_world([x, y, c.slab_t + c.ball_r + 0.001 if z is None else z]))

    def ball_loc():
        return scene._station_local(scene.ball.data.root_pos_w)[0]

    def push_drawer(steps: int, clamp: float = 6.0) -> None:
        """ORACLE actuation of the seed's skill: a PD push on the drawer toward
        the face plane, force-limited to a hand's push. (A real hand cannot even
        reach the drawer front: the guard tunnel interior is 100 mm tall and
        roofed — asserted geometrically in the scene cfg.)"""
        zero = torch.zeros(n, 1, 3, device=device)
        ex = torch.tensor([-1.0, 0.0, 0.0], device=device).expand(n, 3)
        for _ in range(steps):
            axis = quat_apply(scene.station.data.root_quat_w, ex)   # closing dir
            v = (scene.drawer.data.root_lin_vel_w * axis).sum(-1)
            f_mag = (60.0 * scene.drawer_q() - 8.0 * v).clamp(0.0, clamp)
            fw = axis * f_mag.unsqueeze(-1)
            fb = quat_apply_inverse(scene.drawer.data.root_quat_w, fw)
            scene.drawer.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            _step(1)
        scene.drawer.set_external_force_and_torque(zero, zero)
        _step(60)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    bl = ball_loc()
    check("settle/no-NaN: drawer resting at q0 in its channel, ball in its dock "
          "tray, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.drawer_in_channel()[0])
          and not bool(scene.drawer_closed()[0])
          and not bool(scene.ball_delivered_now()[0])
          and not bool(scene.ball_parked()[0])
          and abs(float(scene.drawer_q()[0]) - float(scene.q0[0])) < 0.010
          and abs(float(bl[0]) - c.dock_x) < c.dock_jitter + 0.010
          and abs(float(bl[1]) - c.dock_y) < c.dock_jitter + 0.010
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, q0s, bxy, tracks = [], [], [], [], True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(np.degrees(float(station_yaw()[0])))
        xys.append((scene.station.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        q0s.append(float(scene.q0[0]))
        bl = ball_loc()
        bxy.append([float(bl[0]), float(bl[1])])
        tracks = tracks \
            and abs(float(scene.drawer_q()[0]) - float(scene.q0[0])) < 0.010 \
            and abs(float(bl[0]) - c.dock_x) < c.dock_jitter + 0.008 \
            and abs(float(bl[1]) - c.dock_y) < c.dock_jitter + 0.008 \
            and 0.078 < float(bl[2]) < 0.095      # resting IN the tray, on its floor
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    qspan = max(q0s) - min(q0s)
    bspan = float(np.ptp(np.asarray(bxy), axis=0).max())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} q0span={qspan * 1000:.0f}mm "
          f"dockspan={bspan * 1000:.1f}mm tracks={tracks}", flush=True)
    check("randomization A: station yaw spans > 90 deg and xy jitters across resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: the drawer opening q0 varies and the settled drawer "
          "tracks it; the ball's dock position jitters and the ball rests in the "
          "tray every time",
          qspan > 0.015 and bspan > 0.004 and tracks)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (oracle push on the drawer) ===============================
    torch.manual_seed(41)
    env.reset()
    _step(150)
    push_drawer(360)   # 3 s of a 6 N closing push on the drawer
    _report("seed-skill")
    s5 = float(scene.score()[0])
    bl5 = ball_loc()
    check("SEED strategy: the seed's whole skill (push the drawer shut) executed "
          "by an oracle force — the drawer seats but no ram was ever delivered: "
          "no success, drawer credit only (~0.55)",
          bool(scene.drawer_closed()[0])
          and float(scene._fdeliver[0]) < 0.05
          and abs(float(bl5[0]) - c.dock_x) < 0.03    # ball never left its dock
          and 0.50 <= s5 <= 0.60 and not succ())

    # ================= 6. dead ram (ball laid at the face, no descent) ============================
    torch.manual_seed(51)
    env.reset()
    _step(150)
    q6 = float(scene.drawer_q()[0])
    put_ball(q6 + c.ball_r + 0.002)          # touching the open drawer's front face
    _step(240)                               # no kinetic energy: nothing closes
    _report("dead-ram")
    s6 = float(scene.score()[0])
    q6b = float(scene.drawer_q()[0])
    check("dead ram: the ball laid at rest against the open drawer's face "
          "(transport without the descent) cannot close it — drawer stays put "
          "(< 6 mm), no success, delivery credit only",
          not bool(scene.drawer_closed()[0])
          and abs(q6b - q6) < 0.006
          and float(scene._fdeliver[0]) > 0.95
          and 0.28 <= s6 <= 0.42 and not succ())

    # ================= 7. park near miss (drawer seated, ball short) ==============================
    torch.manual_seed(61)
    env.reset()
    _step(150)
    put_drawer(0.004)                        # seated (inside the 12 mm tolerance)
    put_ball(0.004 + c.ball_r + 0.045)       # gap 45 mm: 20 mm outside park_gap_tol
    _step(240)
    _report("park-miss")
    s7 = float(scene.score()[0])
    check("park near miss: drawer seated + ball at rest on the runway 20 mm "
          "outside the park gap tolerance — parked-clause refuses; base score "
          "capped (<= 0.85 float32), no success",
          bool(scene.drawer_closed()[0]) and not bool(scene.ball_parked()[0])
          and 0.70 <= s7 <= 0.8500002 and not succ())

    # ================= 8. channel guard (drawer stolen, ball delivered) ===========================
    torch.manual_seed(71)
    env.reset()
    _step(150)
    _write_body(scene.drawer, ground(0.95, -0.95, 0.05),
                task_scene._qz(station_yaw()))
    put_ball(0.100)                          # resting on the runway in the tunnel
    _step(240)
    _report("stolen-drawer")
    s8 = float(scene.score()[0])
    check("channel guard: drawer stolen out of the cabinet onto the ground with "
          "the ball resting in the tunnel — the drawer clause and the "
          "drawer-progress latch refuse; delivery credit only, no success",
          not bool(scene.drawer_closed()[0])
          and not bool(scene.drawer_in_channel()[0])
          and float(scene._fdrawer[0]) < 0.05
          and float(scene._fdeliver[0]) > 0.95
          and 0.28 <= s8 <= 0.40 and not succ())

    # ================= 9. latch survives regression ===============================================
    torch.manual_seed(81)
    env.reset()
    _step(150)
    q9 = float(scene.drawer_q()[0])
    put_ball(q9 + c.ball_r + 0.002)          # delivered: latch fires
    _step(60)
    latched = float(scene._fdeliver[0])
    bl9 = ball_loc()
    _write_body(scene.ball, station_world([c.dock_x, c.dock_y,
                                           c.dock_h + c.ball_r + 0.003]))
    _step(120)                               # back in its dock; live reads False
    _report("latch-regress")
    s9 = float(scene.score()[0])
    check("latch survival: ball delivered (rested against the open drawer, latch "
          "fires) then returned to its dock — the delivery latch survives, live "
          "delivered/parked read False, no success",
          latched > 0.95
          and float(scene._fdeliver[0]) > 0.95
          and float(bl9[2]) < 0.08           # it really was down in the tunnel
          and not bool(scene.ball_delivered_now()[0])
          and not bool(scene.ball_parked()[0])
          and s9 >= c.w_deliver - 1e-4
          and not succ())

    # ================= 10-12. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ram_chute_cabinet")
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

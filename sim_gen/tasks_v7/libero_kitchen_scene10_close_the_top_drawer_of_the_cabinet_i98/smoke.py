"""Smoke battery for CamGateCabinetScene (sim_gen task
`libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i98`) — REJECTION-ONLY:
every check either verifies basic health/randomization or CONSTRUCTS a settled
wrong outcome and asserts the rubric refuses it. success() must never fire
anywhere in the battery.

 1. settle/no-NaN     — gate hanging on its pin at theta0, drawer resting at q0,
                        score ~0, no success.
 2. randomization A   — station yaw spans > 90 deg and xy jitters across resets.
 3. randomization B   — the drawer opening q0 AND the gate angle theta0 vary
                        across resets, and the SETTLED state tracks both samples
                        every time (drawer_q / gate_theta readback).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (push the drawer shut) executed
                        for REAL: an 8 N PD push on the drawer front for 3 s. The
                        drawer seats — but the gate still stands open: no
                        success, drawer credit only (~0.45).
 6. both ajar         — near miss: gate constructed at ~10 deg with the drawer at
                        a cam-consistent 20 mm, settled — BOTH clauses refuse
                        (10 > 5 deg, 20 > 12 mm); latched credit stays under the
                        0.90 cap, no success.
 7. channel guard     — drawer STOLEN out of the cabinet onto the ground, gate
                        swung flush on its post: the gate clause holds but the
                        drawer clause and the drawer-progress latch refuse a
                        drawer that is not in its channel; gate credit only.
 8. latch regression  — gate swung to mid-arc (constructed) then returned to its
                        start angle: the gate-progress latch survives, success
                        does not fire, live clauses read open again.
 9. off-hinge fake    — gate lifted OFF its pin and laid flush-ORIENTED on the
                        apron with the drawer seated: gate_on_hinge refuses the
                        gate clause (a gate not on its hinge is not "closed");
                        drawer credit only, no success.
10. rejection audit   — success() observed False at every step of the battery.
11. final no-NaN.
12. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i98.smoke --headless
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
    print(f"[smoke] {tag:18s} | q={float(scene.drawer_q()[0]):+.4f} "
          f"theta={math.degrees(float(scene.gate_theta()[0])):+.1f}deg "
          f"gate_closed={bool(scene.gate_closed()[0])} "
          f"drawer_closed={bool(scene.drawer_closed()[0])} "
          f"on_hinge={bool(scene.gate_on_hinge()[0])} "
          f"in_channel={bool(scene.drawer_in_channel()[0])} "
          f"latch(g/d)=({float(scene._fgate[0]):.2f},{float(scene._fdrawer[0]):.2f}) "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.cam_gate_cabinet")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-1.05, -1.05, 1.00)) + o),
                                tuple(np.array((0.00, 0.00, 0.28)) + o),
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

    def put_gate(theta_rad: float, *, dx: float = 0.0, dz: float = 0.0) -> None:
        """CONSTRUCT: the gate placed on (or, with dx, beside) its hinge pin at
        the given angle. Yaw-only — every construct here is a yaw rotation."""
        _write_body(scene.gate,
                    station_world([c.pin_x + dx, c.pin_y, c.seat_z + 0.001 + dz]),
                    task_scene._qz(station_yaw() + theta_rad))

    def put_drawer(q: float) -> None:
        """CONSTRUCT: the drawer seated in its channel at opening q."""
        _write_body(scene.drawer,
                    station_world([q - c.drawer_l / 2, 0.0, c.slab_t + 0.002]),
                    task_scene._qz(station_yaw()))

    def push_drawer(steps: int, clamp: float = 8.0) -> None:
        """REAL actuation: a PD push on the drawer front toward the face plane
        (the seed's whole skill), force-limited to what a hand would apply."""
        zero = torch.zeros(n, 1, 3, device=device)
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        for _ in range(steps):
            axis = quat_apply(scene.station.data.root_quat_w, ex)
            v = (scene.drawer.data.root_lin_vel_w * axis).sum(-1)
            f_mag = (200.0 * (0.0 - scene.drawer_q()) - 30.0 * v).clamp(-clamp, clamp)
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
    check("settle/no-NaN: gate hanging on its pin at theta0, drawer resting at "
          "q0, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.gate_on_hinge()[0])
          and bool(scene.drawer_in_channel()[0])
          and not bool(scene.gate_closed()[0])
          and not bool(scene.drawer_closed()[0])
          and abs(float(scene.drawer_q()[0]) - float(scene.q0[0])) < 0.010
          and abs(float(scene.gate_theta()[0]) - float(scene.theta0[0]))
          < math.radians(4.0)
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, q0s, th0s, tracks = [], [], [], [], True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(math.degrees(float(station_yaw()[0])))
        xys.append((scene.station.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        q0s.append(float(scene.q0[0]))
        th0s.append(math.degrees(float(scene.theta0[0])))
        tracks = tracks \
            and abs(float(scene.drawer_q()[0]) - float(scene.q0[0])) < 0.010 \
            and abs(float(scene.gate_theta()[0]) - float(scene.theta0[0])) \
            < math.radians(4.0)
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    qspan = max(q0s) - min(q0s)
    tspan = max(th0s) - min(th0s)
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} q0span={qspan * 1000:.0f}mm th0span={tspan:.0f}deg "
          f"tracks={tracks}", flush=True)
    check("randomization A: station yaw spans > 90 deg and xy jitters across resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: the drawer opening q0 AND the gate angle theta0 vary "
          "across resets, and the settled state tracks both samples every time",
          qspan > 0.015 and tspan > 6.0 and tracks)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (real push on the drawer) =================================
    torch.manual_seed(41)
    env.reset()
    _step(150)
    th_before = float(scene.gate_theta()[0])
    push_drawer(360)   # 3 s of an 8 N hand-push on the drawer front
    _report("seed-skill")
    s5 = float(scene.score()[0])
    check("SEED strategy: the seed's whole skill (push the drawer shut) executed "
          "for real — the drawer seats but the gate still stands open: no "
          "success, drawer credit only (~0.45)",
          bool(scene.drawer_closed()[0]) and not bool(scene.gate_closed()[0])
          and abs(float(scene.gate_theta()[0]) - th_before) < math.radians(8.0)
          and 0.40 <= s5 <= 0.55 and not succ())

    # ================= 6. both ajar (near miss under the cap) =====================================
    torch.manual_seed(51)
    env.reset()
    _step(150)
    put_drawer(0.020)                       # 20 mm out: outside the 12 mm tolerance
    put_gate(math.radians(10.0))            # 10 deg: outside the 5 deg tolerance
    _step(240)                              # settle; cam faces stay ~5 mm apart
    _report("both-ajar")
    s6 = float(scene.score()[0])
    check("both ajar: gate at ~10 deg with the drawer at a cam-consistent 20 mm, "
          "settled — both clauses refuse; latched credit stays under the 0.90 "
          "cap, no success",
          not bool(scene.gate_closed()[0]) and not bool(scene.drawer_closed()[0])
          and 0.50 <= s6 <= 0.9000001 and not succ())

    # ================= 7. channel guard (drawer stolen, gate flush) ===============================
    torch.manual_seed(61)
    env.reset()
    _step(150)
    _write_body(scene.drawer, ground(0.90, -0.90, 0.05),
                task_scene._qz(station_yaw()))
    put_gate(0.0)                           # flush on its post — nothing to cam
    _step(300)
    _report("stolen-drawer")
    s7 = float(scene.score()[0])
    check("channel guard: drawer stolen out of the cabinet onto the ground, gate "
          "swung flush — the drawer clause and the drawer-progress latch refuse; "
          "gate credit only, no success",
          bool(scene.gate_closed()[0]) and not bool(scene.drawer_closed()[0])
          and not bool(scene.drawer_in_channel()[0])
          and float(scene._fdrawer[0]) < 0.05
          and 0.40 <= s7 <= 0.52 and not succ())

    # ================= 8. latch survives regression ===============================================
    torch.manual_seed(71)
    env.reset()
    _step(150)
    th0_8 = float(scene.theta0[0])
    th_mid = math.radians(54.0)             # mid-arc, still clear of the widest drawer
    put_gate(th_mid)
    _step(120)
    exp_f = max(0.0, (th0_8 - th_mid) / th0_8)
    latched = float(scene._fgate[0])
    put_gate(th0_8)                         # regression: back to the start angle
    _step(120)
    _report("latch-regress")
    s8 = float(scene.score()[0])
    check("latch regression: gate swung to mid-arc then returned to its start "
          "angle — the gate-progress latch survives, live clauses read open "
          "again, no success",
          latched >= exp_f - 0.03
          and float(scene._fgate[0]) >= latched - 1e-4
          and s8 >= c.w_gate * (exp_f - 0.03) - 1e-6
          and not bool(scene.gate_closed()[0])
          and abs(float(scene.gate_theta()[0]) - th0_8) < math.radians(6.0)
          and not succ())

    # ================= 9. off-hinge fake ==========================================================
    torch.manual_seed(81)
    env.reset()
    _step(150)
    put_drawer(0.005)                       # drawer seated (inside tolerance)
    put_gate(0.0, dx=0.020, dz=-0.008)      # flush-ORIENTED but OFF the pin: rings
    _step(300)                              # beside the pin; panel lands on the apron
    _report("off-hinge")
    s9 = float(scene.score()[0])
    check("off-hinge fake: gate laid flush-oriented on the apron OFF its pin with "
          "the drawer seated — gate_on_hinge refuses the gate clause AND the "
          "gate-progress latch (fgate stays 0 despite the gate transiting "
          "near-flush angles); drawer credit only, no success",
          bool(scene.drawer_closed()[0]) and not bool(scene.gate_on_hinge()[0])
          and not bool(scene.gate_closed()[0])
          and abs(float(scene.gate_theta()[0])) < math.radians(20.0)
          and float(scene._fgate[0]) < 0.05
          and 0.40 <= s9 <= 0.52 and not succ())

    # ================= 10-12. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.cam_gate_cabinet")
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

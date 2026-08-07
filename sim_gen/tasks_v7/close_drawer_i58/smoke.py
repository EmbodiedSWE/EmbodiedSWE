"""Smoke battery for ReturnDockScene (sim_gen task `close_drawer_i58`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS a settled wrong outcome and asserts the rubric refuses it. success()
must never fire anywhere in the battery.

 1. settle/no-NaN     — tray held open on the pin at its socket's opening, bottle
                        aboard, score ~0, no success.
 2. randomization A   — dock yaw spans > 90 deg and xy jitters across resets.
 3. randomization B   — WHICH socket holds the pin varies across resets, and the
                        tray's settled opening tracks that socket every time
                        (face_x readback == openings[socket]).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (push the drawer shut) executed for
                        REAL: an 8 N PD push on the tray for 3 s. The socketed pin
                        arrests it after ~mm; score stays ~0, no success.
 6. header interlock  — pin removed but the bottle left standing in the tray: the
                        tray glides then STALLS when the bottle meets the roof
                        header; stays open, bottle stays aboard, no success.
 7. wrong spot        — path fully cleared but the bottle stood on the GROUND
                        beside the station: tray closes, everything settles, the
                        docked clause refuses; capped partial credit only.
 8. lying on pad      — bottle laid HORIZONTALLY across the pad rim, path clear:
                        tray closes; the upright clause refuses.
 9. pin left standing — bottle docked perfectly but the pin never removed: the
                        tray presses the pin, stays open; no success.
10. pin rides inside  — pin dropped INTO the tray (path now clear): the tray
                        closes with the pin riding along under the header — the
                        pin-clear clause refuses.
11. knocked-over trap — bottle tipped over INSIDE the tray, pin cleared: the
                        fallen bottle passes UNDER the header (the designed trap)
                        and rides into the enclosure — out/docked refuse.
12. latched credit    — bottle docked (constructed) then stolen away to the
                        ground: out+dock latches survive, success does not.
13. rejection audit   — success() observed False at every step of the battery.
14. final no-NaN.
15. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.close_drawer_i58.smoke --headless
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
    print(f"[smoke] {tag:18s} | face_x={float(scene.face_x()[0]):+.4f} "
          f"x0={float(scene.x0[0]):+.3f} socket={int(scene.socket_idx[0])} "
          f"closed={bool(scene.tray_closed()[0])} "
          f"pin_clear={bool(scene.pin_clear()[0])} "
          f"out={bool(scene.bottle_out()[0])} "
          f"docked={bool(scene.bottle_docked()[0])} "
          f"latch(o/d/p)=({int(scene._out[0])},{int(scene._dock_l[0])},"
          f"{int(scene._pin_l[0])}) cf={float(scene._close_f[0]):.2f} "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.return_dock")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((-0.90, -0.90, 0.75)) + o),
                                tuple(np.array((0.00, 0.00, 0.10)) + o),
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

    def dock_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.dock.data.root_pos_w + quat_apply(scene.dock.data.root_quat_w, loc)

    def ground(dx: float, dy: float, h: float) -> torch.Tensor:
        return (scene.env_origins[0:1]
                + torch.tensor([dx, dy, h], device=device)).expand(n, 3)

    def q_dock_mul(axis: str, deg: float) -> torch.Tensor:
        """Dock orientation composed with a rotation about a dock-local axis."""
        half = math.radians(deg) / 2
        v = {"x": (math.sin(half), 0.0, 0.0), "y": (0.0, math.sin(half), 0.0)}[axis]
        qe = torch.tensor([math.cos(half), *v], device=device).expand(n, 4)
        return task_scene._qmul(scene.dock.data.root_quat_w, qe)

    def remove_pin() -> None:
        """CONSTRUCT: the pin stolen away to open ground, upright, well clear."""
        _write_body(scene.pin, ground(0.90, 0.90, c.pin_h / 2 + 0.002))

    def dock_bottle() -> None:
        """CONSTRUCT: the bottle released just above the pad centre; it seats."""
        px, py = c.pad_center
        _write_body(scene.bottle,
                    dock_world([px, py, c.pad_z1 + c.bottle_h / 2 + 0.005]),
                    scene.dock.data.root_quat_w)

    def push_tray(steps: int, clamp: float = 8.0) -> None:
        """REAL actuation: a PD push on the tray toward the doorway (the seed's
        whole skill), force-limited to what a hand would apply."""
        zero = torch.zeros(n, 1, 3, device=device)
        ex = torch.tensor([1.0, 0.0, 0.0], device=device).expand(n, 3)
        for _ in range(steps):
            axis = quat_apply(scene.dock.data.root_quat_w, ex)
            v = (scene.tray.data.root_lin_vel_w * axis).sum(-1)
            f_mag = (200.0 * (0.0 - scene.face_x()) - 30.0 * v).clamp(-clamp, clamp)
            fw = axis * f_mag.unsqueeze(-1)
            fb = quat_apply_inverse(scene.tray.data.root_quat_w, fw)
            scene.tray.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            _step(1)
        scene.tray.set_external_force_and_torque(zero, zero)
        _step(60)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    check("settle/no-NaN: tray held open on the pin at its socket's opening, "
          "bottle aboard, pin seated, score ~0, no success",
          bool(scene._finite()[0]) and not bool(scene.tray_closed()[0])
          and not bool(scene.pin_clear()[0]) and not bool(scene.bottle_out()[0])
          and abs(float(scene.face_x()[0]) - float(scene.x0[0])) < 0.012
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, socks, ok_open = [], [], [], True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)   # tray glides ~4 mm onto the pin
        _refresh()
        yaws.append(yaw_of(scene.dock.data.root_quat_w[0]))
        xys.append((scene.dock.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        socks.append(int(scene.socket_idx[0]))
        ok_open = ok_open and \
            abs(float(scene.face_x()[0]) - float(scene.x0[0])) < 0.012
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} sockets={socks} openings_track={ok_open}", flush=True)
    check("randomization A: dock yaw spans > 90 deg and xy jitters across resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: WHICH socket holds the pin varies across resets, and "
          "the tray's settled opening tracks that socket every time",
          len(set(socks)) >= 2 and ok_open)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (real push on the tray) ===================================
    torch.manual_seed(41)
    env.reset()
    _step(150)
    x0_5 = float(scene.x0[0])
    push_tray(360)   # 3 s of an 8 N hand-push toward the doorway
    _report("seed-skill")
    check("SEED strategy: the seed's whole skill (push the drawer shut) executed "
          "for real — 8 N for 3 s: the socketed pin arrests the tray after ~mm, "
          "score stays ~0, no success",
          float(scene.face_x()[0]) > x0_5 - 0.025
          and float(scene.score()[0]) <= 0.10 and not succ())

    # ================= 6. header interlock (bottle stalls the tray) ===============================
    torch.manual_seed(51)
    env.reset()
    _step(150)
    remove_pin()
    _step(600)   # hands off: tray glides until the standing bottle meets the header
    _report("header-stall")
    check("header interlock: pin removed but the bottle left standing in the tray "
          "— the tray stalls at the roof header, stays open, bottle stays aboard, "
          "no success",
          not bool(scene.tray_closed()[0]) and float(scene.face_x()[0]) > 0.030
          and not bool(scene.bottle_out()[0]) and not succ())

    # ================= 7. wrong spot (bottle on the ground, not the pad) ==========================
    torch.manual_seed(61)
    env.reset()
    _step(150)
    _write_body(scene.bottle, ground(0.90, -0.90, c.bottle_h / 2 + 0.002))
    remove_pin()
    _step(600)   # tray closes; bottle settled upright on open ground
    _report("wrong-spot")
    s7 = float(scene.score()[0])
    check("wrong spot: path fully cleared but the bottle stood on the GROUND "
          "beside the station — tray closes, docked clause refuses; partial "
          "credit only",
          bool(scene.tray_closed()[0]) and bool(scene.pin_clear()[0])
          and not bool(scene.bottle_docked()[0])
          and 0.60 <= s7 <= 0.72 and not succ())

    # ================= 8. lying on pad ============================================================
    torch.manual_seed(71)
    env.reset()
    _step(150)
    px, py = c.pad_center
    _write_body(scene.bottle,
                dock_world([px, py, c.rim_z1 + c.bottle_r + 0.004]),
                q_dock_mul("x", 90.0))   # axis horizontal, lying across the rim
    remove_pin()
    _step(600)
    _report("lying-on-pad")
    check("lying on pad: bottle laid horizontally across the pad rim, path clear "
          "— tray closes; the upright clause refuses",
          bool(scene.tray_closed()[0]) and not bool(scene.bottle_docked()[0])
          and float(scene._bottle_axis_dot()[0])
          < math.cos(math.radians(c.dock_tilt_max_deg)) and not succ())

    # ================= 9. pin left standing =======================================================
    torch.manual_seed(81)
    env.reset()
    _step(150)
    x0_9 = float(scene.x0[0])
    dock_bottle()
    _step(360)   # bottle seats on the pad; tray stays pressed on the pin
    _report("pin-standing")
    s9 = float(scene.score()[0])
    check("pin left standing: bottle docked perfectly but the pin never removed — "
          "the tray stays held open; no success, dock credit only",
          bool(scene.bottle_docked()[0]) and not bool(scene.pin_clear()[0])
          and float(scene.face_x()[0]) > x0_9 - 0.025
          and not bool(scene.tray_closed()[0])
          and 0.35 <= s9 <= 0.50 and not succ())

    # ================= 10. pin rides inside the tray ==============================================
    torch.manual_seed(91)
    env.reset()
    _step(150)
    dock_bottle()
    _step(240)
    # drop the pin INTO the tray (axis along the tray's x: it bridges the walls,
    # low enough to pass under the header) — the runway path is now clear
    tray_q = scene.tray.data.root_quat_w
    tray_c = scene.tray.data.root_pos_w + quat_apply(
        tray_q, torch.tensor([0.0, 0.0, 0.130], device=device).expand(n, 3))
    half = math.radians(90.0) / 2
    q_pin = task_scene._qmul(
        tray_q, torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0],
                             device=device).expand(n, 4))
    _write_body(scene.pin, tray_c, q_pin)
    _step(600)   # pin settles in/on the tray; tray glides shut with it aboard
    _report("pin-rides-in")
    check("pin rides inside: pin dropped into the tray (path now clear) — the "
          "tray closes with the pin riding along; the pin-clear clause refuses",
          bool(scene.tray_closed()[0]) and not bool(scene.pin_clear()[0])
          and not succ())

    # ================= 11. knocked-over trap ======================================================
    torch.manual_seed(101)
    env.reset()
    _step(150)
    # tip the bottle over INSIDE the tray: re-write it leaning ~60 deg from
    # upright with its CENTRE still inside the tray box (no spurious out-latch);
    # it clatters into the lean (base on the floor, shaft on the trailing wall
    # top, tip over the grab bar) which passes UNDER the raised header
    tray_q = scene.tray.data.root_quat_w
    lean_c = scene.tray.data.root_pos_w + quat_apply(
        tray_q, torch.tensor([0.0, 0.0, 0.095], device=device).expand(n, 3))
    half = math.radians(60.0) / 2
    q_lean = task_scene._qmul(
        tray_q, torch.tensor([math.cos(half), 0.0, math.sin(half), 0.0],
                             device=device).expand(n, 4))
    _write_body(scene.bottle, lean_c, q_lean)
    _step(300)   # settle into the lean
    remove_pin()
    _step(600)   # the fallen bottle rides under the header into the enclosure
    _report("trap-lean")
    check("knocked-over trap: bottle tipped over inside the tray, pin cleared — "
          "the fallen bottle passes under the header and rides in; out/docked "
          "refuse success",
          bool(scene.tray_closed()[0]) and not bool(scene.bottle_out()[0])
          and not bool(scene.bottle_docked()[0])
          and not bool(scene._dock_l[0]) and not succ())

    # ================= 12. latched credit =========================================================
    torch.manual_seed(111)
    env.reset()
    _step(150)
    dock_bottle()
    _step(300)
    latched = bool(scene._out[0]) and bool(scene._dock_l[0])
    _write_body(scene.bottle, ground(0.90, -0.90, c.bottle_h / 2 + 0.002))
    _step(180)
    _report("latch")
    s12 = float(scene.score()[0])
    check("latched credit: bottle docked (constructed) then stolen to the ground "
          "— out+dock latches survive, success does not",
          latched and not bool(scene.bottle_docked()[0])
          and 0.38 <= s12 <= 0.50 and not succ())

    # ================= 13-15. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.return_dock")
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

"""Smoke battery for DrawerRerailScene (sim_gen task `close_drawer_i239`) —
REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS a settled wrong outcome and asserts the rubric refuses it. success()
must never fire anywhere in the battery.

 1. settle/no-NaN     — the drawer settles ON ITS SIDE on the apron, score ~0,
                        no success.
 2. randomization A   — cabinet yaw spans > 90 deg and xy jitters across resets.
 3. randomization B   — WHICH bay is marked varies, the beacon's mounting height
                        tracks the marked bay every reset, and WHICH side the
                        drawer lies on varies.
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (push the drawer toward the
                        cabinet) executed for REAL: a force-limited PD shove
                        along the ground. The drawer jams against the plinth
                        face below the mouths; score stays ~0, no success.
 6. decoy bay         — drawer constructed perfectly seated in the UNMARKED bay:
                        physically resting flush, but the marked-bay gating
                        refuses; score ~0.
 7. upside-down       — drawer seated in the MARKED bay but flipped open-top
                        DOWN (it fits — the trap is real): the upright clause
                        refuses; score ~0.
 8. backwards         — drawer inserted handle-first into the MARKED bay until
                        the handle presses the back wall: its outward face
                        stands ~30 mm proud (geometry backs the facing clause);
                        refused, score ~0.
 9. near-miss         — drawer upright/facing/centred in the MARKED bay but left
                        ~35 mm short of flush: no success; capped insertion
                        credit only.
10. on cabinet top    — drawer upright on the cabinet's TOP slab: refused,
                        score ~0.
11. on the ground     — drawer upright on the ground in the marked bay's
                        approach lane: the ground/height gates refuse the hold
                        latch; score ~0, no success.
12. latched credit    — hold+insertion latched (constructed pass through the
                        approach window, then a near-seat), then the drawer
                        stolen away to the ground: partial credit survives,
                        success does not.
13. rejection audit   — success() observed False at every step of the battery.
14. final no-NaN.
15. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.close_drawer_i239.smoke --headless
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
    loc = scene._cab_local(scene.drawer.data.root_pos_w)[0]
    up_dot, face_dot = scene._axes()
    print(f"[smoke] {tag:14s} | front_x={float(scene.front_x()[0]):+.4f} "
          f"loc=({float(loc[0]):+.3f},{float(loc[1]):+.3f},{float(loc[2]):+.3f}) "
          f"up={float(up_dot[0]):+.2f} face={float(face_dot[0]):+.2f} "
          f"bay={int(scene.target_bay[0])} "
          f"seated={bool(scene.seated()[0])} settled={bool(scene.settled()[0])} "
          f"hold={int(scene._hold[0])} ins={float(scene._ins[0]):.2f} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.drawer_rerail")().build(num_envs=args.num_envs, device=device)
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
        env.sim.set_camera_view(tuple(np.array((1.30, -1.30, 0.95)) + o),
                                tuple(np.array((0.00, 0.00, 0.18)) + o),
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

    def cab_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.cabinet.data.root_pos_w + quat_apply(
            scene.cabinet.data.root_quat_w, loc)

    def cab_quat(extra: torch.Tensor | None = None) -> torch.Tensor:
        q = scene.cabinet.data.root_quat_w
        return q if extra is None else task_scene._qmul(q, extra)

    def ground(dx: float, dy: float, h: float) -> torch.Tensor:
        return (scene.env_origins[0:1]
                + torch.tensor([dx, dy, h], device=device)).expand(n, 3)

    def q_ax(fn, deg: float) -> torch.Tensor:
        ang = torch.full((n,), math.radians(deg), device=device)
        return fn(ang)

    def rest_z_of(bay: int) -> float:
        return c.bay_z[bay] + c.d_h / 2

    def marked() -> int:
        return int(scene.target_bay[0])

    def loc0() -> torch.Tensor:
        _refresh()
        return scene._cab_local(scene.drawer.data.root_pos_w)[0]

    def place_in_bay(bay: int, *, centre_x: float, flip: bool = False,
                     backwards: bool = False) -> None:
        """CONSTRUCT: teleport the drawer to a bay-interior pose (fits by the
        asserted clearances: 8 mm/side, 34 mm headroom) and let it settle."""
        extra = None
        if flip:
            extra = q_ax(task_scene._qx, 180.0)      # open top DOWN, handle still out
        if backwards:
            extra = q_ax(task_scene._qz, 180.0)      # handle IN
        _write_body(scene.drawer,
                    cab_world([centre_x, 0.0, rest_z_of(bay) + 0.002]),
                    cab_quat(extra))
        _step(240)

    def push_drawer(steps: int, clamp: float = 2.5) -> None:
        """REAL actuation: a force-limited horizontal PD shove of the fallen
        drawer toward the cabinet front centre (the seed's whole skill, executed
        on the ground)."""
        zero = torch.zeros(n, 1, 3, device=device)
        for _ in range(steps):
            _refresh()
            loc = scene._cab_local(scene.drawer.data.root_pos_w)
            err = torch.zeros(n, 3, device=device)
            err[:, 0] = -loc[:, 0]
            err[:, 1] = -loc[:, 1]
            err_w = quat_apply(scene.cabinet.data.root_quat_w, err)
            err_w[:, 2] = 0.0
            v = scene.drawer.data.root_lin_vel_w.clone()
            v[:, 2] = 0.0
            f_w = 30.0 * err_w - 8.0 * v
            f_norm = f_w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            f_w = f_w * (f_norm.clamp(max=clamp) / f_norm)
            f_b = quat_apply_inverse(scene.drawer.data.root_quat_w, f_w)
            scene.drawer.set_external_force_and_torque(f_b.reshape(n, 1, 3), zero)
            _step(1)
        scene.drawer.set_external_force_and_torque(zero, zero)
        _step(90)

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    up1, _ = scene._axes()
    check("settle/no-NaN: the drawer settles ON ITS SIDE on the apron, "
          "score ~0, no success",
          bool(scene._finite()[0]) and abs(float(up1[0])) < 0.30
          and float(loc0()[2]) < c.d_w / 2 + 0.02
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, bays, sides, beacon_ok, side_ok = [], [], [], [], True, True
    for k in range(10):
        torch.manual_seed(20 + k)
        env.reset()
        _step(30)
        _refresh()
        yaws.append(yaw_of(scene.cabinet.data.root_quat_w[0]))
        xys.append((scene.cabinet.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        b = marked()
        bays.append(b)
        bz = scene._cab_local(scene.beacon.data.root_pos_w)[0]
        beacon_ok = beacon_ok and \
            abs(float(bz[2]) - (c.bay_ceil[b] + c.shelf_t / 2)) < 0.005
        ey = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
        s_side = float(quat_apply(scene.drawer.data.root_quat_w, ey)[0, 2])
        sides.append(1 if s_side > 0 else -1)
        side_ok = side_ok and abs(s_side) > 0.8   # really lying on a side
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} bays={bays} beacon_tracks={beacon_ok} "
          f"sides={sides}", flush=True)
    check("randomization A: cabinet yaw spans > 90 deg and xy jitters across resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: WHICH bay is marked varies, the beacon tracks the "
          "marked bay's mouth every reset, and WHICH side the drawer lies on varies",
          len(set(bays)) >= 2 and beacon_ok and len(set(sides)) >= 2 and side_ok)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> score ~0, no success",
          float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (real shove along the ground) =============================
    torch.manual_seed(41)
    env.reset()
    _step(150)
    push_drawer(420)   # 3.5 s of a 2.5 N shove toward the cabinet front
    _report("seed-skill")
    l5 = loc0()
    check("SEED strategy: the seed's whole skill (push the drawer toward the "
          "cabinet) executed for real — it jams against the plinth below the "
          "mouths; score stays ~0, no success",
          float(l5[0]) > 0.02 and float(l5[2]) < 0.12
          and not bool(scene.seated()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 6. decoy bay ===============================================================
    torch.manual_seed(51)
    env.reset()
    _step(60)
    decoy = 1 - marked()
    place_in_bay(decoy, centre_x=c.seat_x - c.d_l / 2 + 0.001)
    _report("decoy-bay")
    l6 = loc0()
    check("decoy bay: drawer physically seated flush in the UNMARKED bay — the "
          "marked-bay gating refuses; score ~0, no success",
          abs(float(l6[2]) - rest_z_of(decoy)) < 0.02
          and float(scene.front_x()[0]) < c.front_tol
          and not bool(scene.seated()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 7. upside-down in the marked bay ===========================================
    torch.manual_seed(61)
    env.reset()
    _step(60)
    place_in_bay(marked(), centre_x=c.seat_x - c.d_l / 2 + 0.001, flip=True)
    _report("upside-down")
    up7, face7 = scene._axes()
    l7 = loc0()
    check("upside-down: drawer seated in the MARKED bay flipped open-top DOWN "
          "(it fits — the trap is real): the upright clause refuses; score ~0",
          abs(float(l7[2]) - rest_z_of(marked())) < 0.02
          and float(up7[0]) < -0.8 and float(face7[0]) > 0.8
          and not bool(scene.seated()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 8. backwards (handle-first) ================================================
    torch.manual_seed(71)
    env.reset()
    _step(60)
    # handle tip ~2 mm off the back wall: centre = -bay_d + gap + handle_len + d_l/2
    bx = -c.bay_d + 0.002 + c.handle_len + c.d_l / 2
    place_in_bay(marked(), centre_x=bx, backwards=True)
    _report("backwards")
    _, face8 = scene._axes()
    l8 = loc0()
    outer_face = float(l8[0]) + c.d_l / 2   # the face pointing OUT of the bay
    check("backwards: drawer inserted handle-first until the handle presses the "
          "back wall — its outward face stands proud beyond the flush window "
          "(geometry backs the facing clause); refused, score ~0",
          float(face8[0]) < -0.8 and outer_face > c.front_tol + 0.010
          and not bool(scene.seated()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 9. near-miss (short of flush) ==============================================
    torch.manual_seed(81)
    env.reset()
    _step(60)
    place_in_bay(marked(), centre_x=0.035 - c.d_l / 2)   # front face ~35 mm proud
    _report("near-miss")
    s9 = float(scene.score()[0])
    check("near-miss: drawer upright/facing/centred in the MARKED bay but left "
          "~35 mm short of flush — no success; capped insertion credit only",
          float(scene.front_x()[0]) > c.front_tol + 0.005
          and not bool(scene.seated()[0])
          and 0.30 <= s9 <= 0.55 and not succ())

    # ================= 10. on the cabinet top =====================================================
    torch.manual_seed(91)
    env.reset()
    _step(60)
    z_top = c.bay_ceil[1] + c.top_t
    _write_body(scene.drawer,
                cab_world([-c.bay_d / 2, 0.0, z_top + c.d_h / 2 + 0.002]),
                cab_quat())
    _step(240)
    _report("on-top")
    l10 = loc0()
    check("on cabinet top: drawer upright on the TOP slab — refused, score ~0",
          float(l10[2]) > z_top and not bool(scene.seated()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 11. upright on the ground in the approach lane =============================
    torch.manual_seed(101)
    env.reset()
    _step(60)
    _write_body(scene.drawer,
                cab_world([0.15, 0.0, c.d_h / 2 + 0.002]),
                cab_quat())
    _step(240)
    _report("on-ground")
    up11, face11 = scene._axes()
    check("on the ground: drawer upright, facing out, centred in the approach "
          "lane but ON THE GROUND — the ground/height gates refuse the hold "
          "latch; score ~0, no success",
          float(up11[0]) > 0.9 and float(face11[0]) > 0.9
          and not bool(scene._hold[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 12. latched credit survives theft ==========================================
    torch.manual_seed(111)
    env.reset()
    _step(60)
    rz = rest_z_of(marked())
    # pass through the approach window (hold latches while it hovers/falls) ...
    _write_body(scene.drawer, cab_world([0.15, 0.0, rz + 0.005]), cab_quat())
    _step(10)
    hold12 = bool(scene._hold[0])
    # ... then a near-seat (insertion latches ~0.79) ...
    place_in_bay(marked(), centre_x=0.035 - c.d_l / 2)
    ins12 = float(scene._ins[0])
    # ... then the drawer is STOLEN away to open ground.
    _write_body(scene.drawer, ground(0.60, -0.35, c.d_h / 2 + 0.002), cab_quat())
    _step(180)
    _report("theft")
    s12 = float(scene.score()[0])
    check("latched credit: hold + insertion latched, then the drawer stolen to "
          "the ground — partial credit survives, success does not",
          hold12 and 0.70 <= ins12 <= 0.85
          and 0.45 <= s12 <= 0.66 and not succ())

    # ================= 13-15. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.drawer_rerail")
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

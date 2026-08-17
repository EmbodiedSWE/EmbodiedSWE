"""Smoke battery for WeighPressCabinetScene (sim_gen task
`libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_i323`)
— REJECTION-ONLY: every check either verifies basic health/randomization or
CONSTRUCTS a settled wrong outcome and asserts the rubric refuses it. success()
must never fire anywhere in the battery.

 1. settle/no-NaN     — drawer parked OPEN at the derived q_open, blade flush on the
                        fin, bowl on its dealt plinth slot, score ~0, no success.
 2. randomization A   — station yaw spans > 90 deg and xy jitters across resets.
 3. randomization B   — the bowl's dealt slot permutes (>= 2 distinct), the bowl
                        tracks its slot with real jitter, the drawer rest tracks
                        q_open and the plunger rests flush — every reset (readback).
 4. null policy       — 240 idle steps -> drawer stays open, score ~0, no success.
 5. SEED strategy     — the seed's WHOLE end state constructed seed-style: the bowl
                        set on the BARE cabinet roof AND the drawer pushed shut by
                        an ORACLE hand-force (PD, <= 9 N). Judged WHILE HELD: both
                        seed clauses look satisfied yet success() is False (bowl not
                        in the tray). Released: the dead-man drawer re-opens by
                        itself. Closure credit only (~0.45), no success.
 6. light load        — both butter boxes set INTO the weigh tray: 40 g is below
                        the drive threshold, the drawer does not move, score ~0.
 7. inverted bowl     — the bowl set UPSIDE-DOWN into the tray: its weight DOES
                        close the drawer (machine works) but the upright clause
                        refuses; end state differs from success only by the bowl's
                        orientation — no success, closure credit only (~0.45).
 8. hand-slam reopen  — drawer + plunger teleport-written to the seated pose (the
                        latch fires, proving the probe touched the goal): with no
                        weight in the tray the wedge back-drives and the drawer
                        re-opens on its own — no settled closed state exists.
 9. latch yank        — bowl dropped into the tray (load latch fires mid-press),
                        then YANKED to the ground: the latch survives but the live
                        clauses read False and the drawer re-opens — no success.
10. shaft guard       — plunger STOLEN to the ground, drawer slammed shut, bowl
                        parked on the roof over the machine: bowl-on-tray requires
                        the plunger in its shaft — refused, drawer re-opens.
11. rejection audit   — success() observed False at every step of the battery.
12. final no-NaN.
13. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_i323.smoke --headless
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
    bl = scene._station_local(scene.bowl.data.root_pos_w)[0]
    pl = scene._station_local(scene.plunger.data.root_pos_w)[0]
    print(f"[smoke] {tag:14s} | q={float(scene.drawer_q()[0]):+.4f} "
          f"plunger_z={float(pl[2]):+.4f} "
          f"bowl_loc=({float(bl[0]):+.3f},{float(bl[1]):+.3f},{float(bl[2]):+.3f}) "
          f"on_tray={bool(scene.bowl_on_tray()[0])} "
          f"in_shaft={bool(scene.plunger_in_shaft()[0])} "
          f"closed={bool(scene.drawer_closed()[0])} "
          f"in_channel={bool(scene.drawer_in_channel()[0])} "
          f"latch(ld/cl)=({float(scene._fload[0]):.2f},{float(scene._fclose[0]):.2f}) "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.weigh_press_cabinet")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-1.30, -1.45, 1.30)) + o),
                                tuple(np.array((0.10, 0.00, 0.42)) + o),
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

    def st_quat() -> torch.Tensor:
        return scene.station.data.root_quat_w

    def station_yaw() -> torch.Tensor:
        return task_scene._yaw_of(st_quat())

    def station_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.station.data.root_pos_w + quat_apply(st_quat(), loc)

    def plunger_rel(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.plunger.data.root_pos_w + quat_apply(st_quat(), loc)

    def ground(dx: float, dy: float, h: float) -> torch.Tensor:
        return (scene.env_origins[0:1]
                + torch.tensor([dx, dy, h], device=device)).expand(n, 3)

    def bowl_loc():
        return scene._station_local(scene.bowl.data.root_pos_w)[0]

    def plunger_loc():
        return scene._station_local(scene.plunger.data.root_pos_w)[0]

    def put_drawer(q: float) -> None:
        """CONSTRUCT: the drawer seated on its slideway at opening q."""
        quat = quat_mul(st_quat(),
                        task_scene._qy_t(math.radians(c.pitch_deg), n, device))
        _write_body(scene.drawer, station_world([q, 0.0, c.z_run(q) + 0.002]), quat)

    def put_plunger_flush(q: float) -> None:
        """CONSTRUCT: the plunger on its shaft axis, blade flush on the fin at
        drawer opening q (write the WHOLE linkage — never one body of a wedge)."""
        _write_body(scene.plunger,
                    station_world([c.blade_x, 0.0, c.z_press(q) + 0.002]),
                    st_quat().clone())

    def push_drawer(steps: int, clamp: float = 9.0) -> tuple[bool, bool]:
        """ORACLE actuation of the seed's skill: a PD hand-push on the drawer
        toward the face plane (feed-forward beats slope gravity + friction, clamp
        is a light hand's push). Returns (drawer_closed sampled WHILE HELD,
        success sampled WHILE HELD) — the probe proves it really seated it."""
        zero = torch.zeros(n, 1, 3, device=device)
        ex = torch.tensor([-1.0, 0.0, 0.0], device=device).expand(n, 3)
        held_closed = False
        held_success = False
        for _ in range(steps):
            axis = quat_apply(st_quat(), ex)                # closing direction
            v = (scene.drawer.data.root_lin_vel_w * axis).sum(-1)
            f_mag = (5.5 + 60.0 * scene.drawer_q() - 8.0 * v).clamp(0.0, clamp)
            fw = axis * f_mag.unsqueeze(-1)
            fb = quat_apply_inverse(scene.drawer.data.root_quat_w, fw)
            scene.drawer.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            _step(1)
            held_closed = held_closed or bool(scene.drawer_closed()[0])
            held_success = held_success or succ()
        scene.drawer.set_external_force_and_torque(zero, zero)
        return held_closed, held_success

    def dealt_slot_xy() -> tuple[float, float]:
        return c.slots[int(scene.bowl_slot[0])]

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    bl, pl = bowl_loc(), plunger_loc()
    q1 = float(scene.drawer_q()[0])
    sx, sy = dealt_slot_xy()
    check("settle/no-NaN: drawer parked OPEN at q_open, blade flush on the fin, "
          "bowl on its dealt plinth slot, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.drawer_in_channel()[0])
          and not bool(scene.drawer_closed()[0])
          and abs(q1 - c.q_open) < 0.008
          and bool(scene.plunger_in_shaft()[0])
          and abs(float(pl[2]) - c.z_press(q1)) < 0.010
          and abs(float(bl[0]) - sx) < c.slot_jitter + 0.015
          and abs(float(bl[1]) - sy) < c.slot_jitter + 0.015
          and -0.010 < float(bl[2]) < 0.020
          and not bool(scene.bowl_on_tray()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, slots, jit, tracks = [], [], [], [], True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(80)
        _refresh()
        yaws.append(np.degrees(float(station_yaw()[0])))
        xys.append((scene.station.data.root_pos_w[0]
                    - scene.env_origins[0])[:2].tolist())
        slots.append(int(scene.bowl_slot[0]))
        bl, pl = bowl_loc(), plunger_loc()
        sx, sy = dealt_slot_xy()
        jit.append([float(bl[0]) - sx, float(bl[1]) - sy])
        qk = float(scene.drawer_q()[0])
        tracks = tracks \
            and abs(qk - c.q_open) < 0.008 \
            and abs(float(pl[2]) - c.z_press(qk)) < 0.010 \
            and abs(float(bl[0]) - sx) < c.slot_jitter + 0.015 \
            and abs(float(bl[1]) - sy) < c.slot_jitter + 0.015 \
            and -0.010 < float(bl[2]) < 0.020
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    jspan = float(np.ptp(np.asarray(jit), axis=0).max())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} slots={slots} jitspan={jspan * 1000:.1f}mm "
          f"tracks={tracks}", flush=True)
    check("randomization A: station yaw spans > 90 deg and xy jitters across resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: the bowl's dealt slot permutes (>= 2 distinct) with "
          "real in-slot jitter; the drawer rest tracks q_open and the plunger "
          "rests flush on the fin every reset",
          len(set(slots)) >= 2 and jspan > 0.004 and tracks)

    # ================= 4. null policy =============================================================
    torch.manual_seed(31)
    env.reset()
    _step(240)
    _report("null")
    check("null policy: 240 idle steps -> drawer stays open, score ~0, no success",
          float(scene.drawer_q()[0]) > 0.045
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 5. SEED strategy (bowl on bare roof + oracle hand-push) ====================
    torch.manual_seed(41)
    env.reset()
    _step(150)
    # the seed's second clause, seed-style: bowl parked on the BARE cabinet roof
    # (the strip in front of the tray) — genuinely "on top of the cabinet"
    _write_body(scene.bowl, station_world([-0.010, 0.0, c.roof_z1 + 0.004]),
                st_quat().clone())
    _step(60)
    bl5 = bowl_loc()
    on_roof = abs(float(bl5[0]) + 0.010) < 0.030 and \
        c.roof_z1 - 0.010 < float(bl5[2]) < c.roof_z1 + 0.030
    # the seed's first clause, seed-style: push the drawer shut by hand (oracle)
    held_closed, held_success = push_drawer(360)
    _report("seed-held")
    _step(300)                               # hand off: the dead-man wedge back-drives
    _report("seed-released")
    s5 = float(scene.score()[0])
    check("SEED strategy: bowl on the bare roof AND the drawer pushed shut by an "
          "oracle hand-force — WHILE HELD both seed clauses look done yet success "
          "is False (bowl not in the tray); released, the drawer re-opens by "
          "itself; closure credit only (~0.45), no success",
          on_roof and held_closed and not held_success
          and float(scene.drawer_q()[0]) > 0.045
          and float(scene._fload[0]) < 0.05
          and 0.42 <= s5 <= 0.47 and not succ())

    # ================= 6. light load (butter in the tray, below threshold) ========================
    torch.manual_seed(51)
    env.reset()
    _step(150)
    q6 = float(scene.drawer_q()[0])
    _write_body(scene.butter1, plunger_rel([0.0, 0.028, c.tray_z1 + 0.010]),
                st_quat().clone())
    _write_body(scene.butter2, plunger_rel([0.0, -0.028, c.tray_z1 + 0.010]),
                st_quat().clone())
    _step(240)
    _report("butter-load")
    check("light load: both butter boxes (40 g) set INTO the weigh tray are below "
          "the drive threshold — the drawer does not move, score ~0, no success",
          abs(float(scene.drawer_q()[0]) - q6) < 0.008
          and not bool(scene.drawer_closed()[0])
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 7. inverted bowl (machine works, upright clause refuses) ===================
    torch.manual_seed(61)
    env.reset()
    _step(150)
    flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(n, 4)
    _write_body(scene.bowl,
                plunger_rel([0.0, 0.0, c.tray_z1 + c.bowl_h + 0.006]),
                quat_mul(st_quat(), flip))
    _step(360)
    _report("inverted")
    s7 = float(scene.score()[0])
    check("inverted bowl: set UPSIDE-DOWN into the tray its weight DOES close the "
          "drawer, but the upright clause refuses — the end state differs from "
          "success only by the bowl's orientation; no success, closure credit "
          "only (~0.45)",
          bool(scene.drawer_closed()[0])
          and not bool(scene.bowl_on_tray()[0])
          and float(scene._fload[0]) < 0.05
          and 0.42 <= s7 <= 0.47 and not succ())

    # ================= 8. hand-slam reopen (dead-man) =============================================
    torch.manual_seed(71)
    env.reset()
    _step(150)
    put_drawer(0.004)                            # 1 mm off the rear stop: f ~ 0.98
    put_plunger_flush(0.004)
    slammed = bool(scene.drawer_closed()[0])     # the probe really touched the goal
    _step(420)                                   # hands off: back-drive + reopen
    _report("slam-reopen")
    check("hand-slam reopen: drawer + plunger written to the seated pose (closure "
          "latch fires — the probe is not vacuous): with the tray empty the wedge "
          "back-drives and the drawer re-opens on its own; no settled closed "
          "state, no success",
          slammed and float(scene._fclose[0]) > 0.95
          and float(scene.drawer_q()[0]) > 0.045
          and not bool(scene.drawer_closed()[0])
          and bool(scene.plunger_in_shaft()[0])
          and not succ())

    # ================= 9. latch survives the yank =================================================
    torch.manual_seed(81)
    env.reset()
    _step(150)
    _write_body(scene.bowl, plunger_rel([0.0, 0.0, c.tray_z1 + 0.008]),
                st_quat().clone())
    _step(10)                                    # lands in the tray: load latch fires
    loaded = float(scene._fload[0])
    _write_body(scene.bowl, ground(1.0, 0.0, 0.06))
    _step(300)                                   # unloaded: the press lets go
    _report("latch-yank")
    s9 = float(scene.score()[0])
    check("latch yank: bowl dropped into the tray (load latch fires) then yanked "
          "to the ground — the latch survives but the live clauses read False and "
          "the drawer re-opens; partial credit only, no success",
          loaded > 0.95 and float(scene._fload[0]) > 0.95
          and not bool(scene.bowl_on_tray()[0])
          and float(scene.drawer_q()[0]) > 0.045
          and 0.30 <= s9 <= 0.7500002 and not succ())

    # ================= 10. shaft guard (plunger stolen) ===========================================
    torch.manual_seed(91)
    env.reset()
    _step(150)
    _write_body(scene.plunger, ground(1.0, 0.9, 0.10))
    put_drawer(0.004)
    _write_body(scene.bowl, station_world([-0.060, 0.0, c.roof_z1 + 0.004]),
                st_quat().clone())
    _step(420)
    _report("stolen-shaft")
    check("shaft guard: plunger stolen to the ground, drawer slammed shut, bowl "
          "parked on the roof over the machine — bowl-on-tray requires the "
          "plunger in its shaft (refused) and the unheld drawer re-opens; "
          "no success",
          not bool(scene.plunger_in_shaft()[0])
          and not bool(scene.bowl_on_tray()[0])
          and float(scene._fload[0]) < 0.05
          and float(scene.drawer_q()[0]) > 0.045
          and not succ())

    # ================= 11-13. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.weigh_press_cabinet")
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

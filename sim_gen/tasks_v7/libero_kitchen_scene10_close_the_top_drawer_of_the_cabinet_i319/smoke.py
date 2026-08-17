"""Smoke battery for BallastPressCabinetScene (sim_gen task
`libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i319`) — REJECTION-ONLY:
every check either verifies basic health/randomization or CONSTRUCTS a settled wrong
outcome and asserts the rubric refuses it. success() must never fire anywhere in the
battery.

 1. settle/no-NaN     — drawer resting at its sampled opening q0 in its channel, lever
                        up on its stop, all three blocks docked on the deck strips,
                        score ~0, no success.
 2. randomization A   — station yaw spans > 90 deg and xy jitters across resets.
 3. randomization B   — the drawer opening q0 varies across resets and the settled
                        drawer tracks it; the block dock layout varies and every block
                        rests ON the deck inside its strip (readback).
 4. null policy       — 240 idle steps -> score ~0, no success.
 5. SEED strategy     — the seed's whole skill (push the drawer shut by hand) executed
                        by an ORACLE force: a 2.5 N closing push seats the drawer for
                        real. Hopper empty, lever up: no success, drawer credit only
                        (~0.55).
 6. one-block threshold — ONE block set in a hopper cell: provably below the press
                        threshold (the lever does NOT move, the drawer does NOT move);
                        one ballast credit only (~0.15), no success.
 7. hand-press        — the lever heeled past the press angle by an ORACLE torque
                        (hopper empty): the blade seats the drawer, but the moment the
                        hand lets go the counterweight swings the lever back UP — the
                        press does not persist without ballast; drawer credit only
                        (~0.55), no success at any settled point.
 8. combined near miss — drawer seated + ONE block riding a cell (two clauses of
                        three): lever still up, block count 1 < 2 -> no success,
                        ~0.70.
 9. channel guard     — drawer STOLEN out of its channel and parked on the cabinet
                        roof with its face x reading "closed": the channel guard
                        refuses the closure clause AND the closure latch; no drawer
                        credit, no success.
10. latch survival    — a block set in a cell (latch fires after its residency
                        streak) then RETURNED to the deck: the ballast latch survives,
                        the live in-cell count reads 0, no success.
11. rejection audit   — success() observed False at every step of the battery.
12. final no-NaN.
13. video             — frames.npz (>10 frames) written to the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_close_the_top_drawer_of_the_cabinet_i319.smoke --headless
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
    inc = scene.blocks_in_cells()[0]
    print(f"[smoke] {tag:16s} | q={float(scene.drawer_q()[0]):+.4f} "
          f"theta={float(scene.lever_theta_deg()[0]):+.2f}deg "
          f"in_cells={[bool(v) for v in inc]} "
          f"closed={bool(scene.drawer_closed()[0])} "
          f"in_channel={bool(scene.drawer_in_channel()[0])} "
          f"latch(blk/dr)=({float(scene._fblock[0].sum()):.1f},{float(scene._fdrawer[0]):.2f}) "
          f"settled={bool(scene.settled()[0])} "
          f"score={float(scene.score()[0]):.3f} "
          f"success={bool(scene.success()[0])} frames={len(_REC['frames'])}",
          flush=True)


# ----- main ----------------------------------------------------------------------------------------
def main() -> None:
    global _ENV
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.ballast_press_cabinet")().build(num_envs=args.num_envs, device=device)
    _ENV = env
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    from isaaclab.utils.math import quat_apply

    # --- recording (viewport rgb annotator) ---
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-1.30, -1.30, 1.15)) + o),
                                tuple(np.array((0.15, 0.00, 0.40)) + o),
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
        return task_scene._yaw_of(scene.base.data.root_quat_w)

    def station_world(local_xyz) -> torch.Tensor:
        loc = torch.tensor(local_xyz, device=device, dtype=torch.float).expand(n, 3)
        return scene.base.data.root_pos_w + quat_apply(scene.base.data.root_quat_w, loc)

    def block_loc(k: int) -> torch.Tensor:
        return scene._station_local(scene.blocks[k].data.root_pos_w)[0]

    def blocks_docked() -> bool:
        """All three blocks resting ON the deck inside their dock strips (readback)."""
        ok = True
        for k in range(3):
            d = block_loc(k)
            ok = ok and c.dock_x_range[0] - 0.02 < float(d[0]) < c.dock_x_range[1] + 0.02 \
                and c.dock_y_range[0] - 0.02 < abs(float(d[1])) < c.dock_y_range[1] + 0.02 \
                and abs(float(d[2]) - c.plinth_h - c.cube_s / 2) < 0.010
        return ok

    def put_drawer(q: float) -> None:
        """CONSTRUCT: the drawer riding in its channel at opening q."""
        _write_body(scene.drawer,
                    station_world([q - c.drawer_l / 2, 0.0, c.plinth_h + c.slab_t + 0.002]),
                    scene.base.data.root_quat_w)

    def put_block_cell(k: int, cell_y: float) -> None:
        """CONSTRUCT: block k set gently INTO a hopper cell (3 mm above its floor),
        mapped through the lever's measured pose — same transport the solve performs."""
        loc = torch.zeros(n, 3, device=device)
        loc[:, 0] = (c.tray_x0 + c.tray_x1) / 2
        loc[:, 1] = cell_y
        loc[:, 2] = 0.003
        _write_body(scene.blocks[k],
                    scene.lever.data.root_pos_w
                    + quat_apply(scene.lever.data.root_quat_w, loc),
                    scene.lever.data.root_quat_w)

    def put_block_deck(k: int, x: float, y: float) -> None:
        _write_body(scene.blocks[k],
                    station_world([x, y, c.plinth_h + c.cube_s / 2 + 0.002]),
                    scene.base.data.root_quat_w)

    def push_drawer(steps: int, f_close: float = 2.5) -> None:
        """ORACLE actuation of the seed's whole skill: a constant closing push on the
        drawer along the station's -x (a hand's push, ~2.5 N)."""
        zero = torch.zeros(n, 1, 3, device=device)
        from isaaclab.utils.math import quat_apply_inverse

        ex = torch.tensor([-1.0, 0.0, 0.0], device=device).expand(n, 3)
        for _ in range(steps):
            fw = quat_apply(scene.base.data.root_quat_w, ex) * f_close
            fb = quat_apply_inverse(scene.drawer.data.root_quat_w, fw)
            scene.drawer.set_external_force_and_torque(fb.reshape(n, 1, 3), zero)
            _step(1)
        scene.drawer.set_external_force_and_torque(zero, zero)
        _step(90)

    def press_lever(steps: int, tau: float = 4.0) -> None:
        """ORACLE hand-press on the lever: a heel-ward torque about its hinge axis
        (body +y), strong enough to beat the empty counterweight."""
        zero = torch.zeros(n, 1, 3, device=device)
        tq = torch.tensor([0.0, tau, 0.0], device=device).expand(n, 1, 3)
        for _ in range(steps):
            scene.lever.set_external_force_and_torque(zero, tq)
            _step(1)

    def release_lever(steps: int) -> None:
        zero = torch.zeros(n, 1, 3, device=device)
        scene.lever.set_external_force_and_torque(zero, zero)
        _step(steps)

    cy = c.cell_centers_y()

    # ================= 1. settle / no-NaN =========================================================
    torch.manual_seed(11)
    env.reset()
    _step(150)
    _report("reset")
    check("settle/no-NaN: drawer resting at q0 in its channel, lever up on its stop, "
          "all three blocks docked on the deck, score ~0, no success",
          bool(scene._finite()[0]) and bool(scene.drawer_in_channel()[0])
          and not bool(scene.drawer_closed()[0])
          and abs(float(scene.drawer_q()[0]) - float(scene.q0[0])) < 0.010
          and abs(float(scene.lever_theta_deg()[0])) < 1.5
          and not bool(scene.blocks_in_cells()[0].any())
          and blocks_docked()
          and float(scene.score()[0]) <= 0.05 and not succ())

    # ================= 2+3. randomization readback ================================================
    yaws, xys, q0s, dock0, tracks = [], [], [], [], True
    for k in range(8):
        torch.manual_seed(20 + k)
        env.reset()
        _step(60)
        _refresh()
        yaws.append(np.degrees(float(station_yaw()[0])))
        xys.append((scene.base.data.root_pos_w[0] - scene.env_origins[0])[:2].tolist())
        q0s.append(float(scene.q0[0]))
        d = block_loc(0)
        dock0.append([float(d[0]), float(d[1])])
        tracks = tracks \
            and abs(float(scene.drawer_q()[0]) - float(scene.q0[0])) < 0.010 \
            and blocks_docked()
    yspan = max(yaws) - min(yaws)
    xystd = float(np.std(np.asarray(xys), axis=0).mean())
    qspan = max(q0s) - min(q0s)
    dspan = float(np.ptp(np.asarray(dock0), axis=0).max())
    print(f"[smoke] readback: yaws={[f'{y:+.0f}' for y in yaws]} span={yspan:.0f} "
          f"xystd={xystd:.3f} q0span={qspan * 1000:.0f}mm "
          f"dockspan={dspan * 1000:.0f}mm tracks={tracks}", flush=True)
    check("randomization A: station yaw spans > 90 deg and xy jitters across resets",
          yspan > 90.0 and xystd > 0.008)
    check("randomization B: the drawer opening q0 varies and the settled drawer "
          "tracks it; the block dock layout varies and every block rests on the "
          "deck inside its strip every reset",
          qspan > 0.015 and dspan > 0.020 and tracks)

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
    push_drawer(300)   # 2.5 s of a 2.5 N closing push -> the drawer really seats
    _report("seed-skill")
    s5 = float(scene.score()[0])
    check("SEED strategy: the seed's whole skill (push the drawer shut) executed by "
          "an oracle force — the drawer seats for real, but the hopper is empty and "
          "the lever never moved: drawer credit only (~0.55), no success",
          bool(scene.drawer_closed()[0])
          and abs(float(scene.lever_theta_deg()[0])) < 2.0
          and float(scene._fblock[0].sum()) < 0.5
          and 0.50 <= s5 <= 0.60 and not succ())

    # ================= 6. one-block threshold =====================================================
    torch.manual_seed(51)
    env.reset()
    _step(150)
    q6 = float(scene.drawer_q()[0])
    put_block_cell(0, cy[1])                 # one block, centre cell
    _step(300)
    _report("one-block")
    s6 = float(scene.score()[0])
    check("one-block threshold: ONE block set in a hopper cell is provably below the "
          "press threshold — the lever stays up, the drawer does not move; one "
          "ballast credit only (~0.15), no success",
          bool(scene.blocks_in_cells()[0, 0])
          and float(scene.lever_theta_deg()[0]) < 2.0
          and abs(float(scene.drawer_q()[0]) - q6) < 0.005
          and 0.13 <= s6 <= 0.22 and not succ())

    # ================= 7. hand-press (oracle torque, hopper empty) ================================
    torch.manual_seed(61)
    env.reset()
    _step(150)
    press_lever(300)                         # heel the lever by hand: blade seats the drawer
    _refresh()
    th_pressed = float(scene.lever_theta_deg()[0])
    closed_pressed = bool(scene.drawer_closed()[0])
    release_lever(480)                       # let go: the counterweight takes over
    _report("hand-press")
    s7 = float(scene.score()[0])
    check("hand-press: an oracle torque heels the lever past the press angle and the "
          "blade seats the drawer — but on release the counterweight swings the lever "
          "back UP (the press cannot persist without ballast): drawer credit only "
          "(~0.55), no success",
          th_pressed >= c.press_min_deg and closed_pressed
          and float(scene.lever_theta_deg()[0]) < 5.0
          and 0.50 <= s7 <= 0.60 and not succ())

    # ================= 8. combined near miss (2 of 3 clauses) =====================================
    torch.manual_seed(71)
    env.reset()
    _step(150)
    put_drawer(0.004)                        # drawer seated (inside the 12 mm tolerance)
    _step(60)
    put_block_cell(1, cy[2])                 # ONE block riding the +y cell
    _step(300)
    _report("near-miss-2of3")
    s8 = float(scene.score()[0])
    check("combined near miss: drawer seated + ONE block riding a cell — the lever "
          "is still up and the in-cell count is 1 < 2: ~0.70, no success",
          bool(scene.drawer_closed()[0])
          and bool(scene.blocks_in_cells()[0, 1])
          and float(scene.lever_theta_deg()[0]) < 2.0
          and 0.63 <= s8 <= 0.745 and not succ())

    # ================= 9. channel guard (drawer stolen onto the roof) =============================
    torch.manual_seed(81)
    env.reset()
    _step(150)
    # parked on the cabinet roof: its face x READS "closed" but it is not in its channel
    _write_body(scene.drawer,
                station_world([0.005 - c.drawer_l / 2, 0.0, c.plinth_h + 0.152]),
                scene.base.data.root_quat_w)
    _step(240)
    _report("stolen-drawer")
    s9 = float(scene.score()[0])
    lq = scene._station_local(scene.drawer.data.root_pos_w)[0]
    check("channel guard: drawer stolen out of its channel and parked on the cabinet "
          "roof with its face x reading closed — the guard refuses the closure clause "
          "and the closure latch: no drawer credit, no success",
          float(lq[0]) + c.drawer_l / 2 < 0.020
          and not bool(scene.drawer_in_channel()[0])
          and not bool(scene.drawer_closed()[0])
          and float(scene._fdrawer[0]) < 0.05
          and s9 <= 0.05 and not succ())

    # ================= 10. latch survival =========================================================
    torch.manual_seed(91)
    env.reset()
    _step(150)
    put_block_cell(2, cy[0])                 # block C into the -y cell
    _step(120)                               # residency streak -> latch fires
    latched = float(scene._fblock[0, 2])
    put_block_deck(2, 0.30, 0.20)            # ...and back out onto the deck
    _step(120)
    _report("latch-regress")
    s10 = float(scene.score()[0])
    check("latch survival: a block set in a cell (latch fires after its residency "
          "streak) then returned to the deck — the ballast latch survives, the live "
          "in-cell count reads 0, no success",
          latched > 0.5
          and float(scene._fblock[0, 2]) > 0.5
          and not bool(scene.blocks_in_cells()[0].any())
          and s10 >= c.w_block - 1e-4
          and not succ())

    # ================= 11-13. audit, no-NaN, video ================================================
    check("rejection audit: success() observed False at every step of the battery",
          not _AUDIT["saw_success"])
    check("final no-NaN", bool(scene._finite()[0]))

    if _REC["frames"]:
        arr = np.stack(_REC["frames"], axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.ballast_press_cabinet")
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

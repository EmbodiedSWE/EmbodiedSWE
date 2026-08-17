"""Smoke battery for RockerTwinDrawersScene — REJECTION tests for the rubric,
NullRobot, RECORDED.

This is NOT a solution (the solution is solve.py — push the open twin drawer fully
in so the hidden walking beam slides the shut GOAL drawer out, drop the pudding
into the exposed tray, push the goal drawer shut; the Franka strategy is TASK.md's
embodiment argument). Teleported states here are rubric INSTRUMENTATION: construct
an outcome as a settled state under the scene's live plants, then assert the
rubric's verdict.

A fail-fast GEOMETRY AUDIT runs first (pure cfg arithmetic, no sim): panel flush
with the chest face at ext 0, aperture clearances > summed contact offsets,
pin-fork engagement across the whole travel, 4 mm backlash, roof/wall gaps smaller
than the pudding (no vertical or over-wall entry), exposed open tray longer than
the pudding.

One linear run, 11 named checks:
  1. settle    — clean reset: finite state, goal drawer shut flush / twin fully out
                 (READBACK), pudding on the floor in its spawn band, score ~0;
  2. random    — goal SIDE flips across seeds (both sides seen) and pudding xy/yaw
                 draws differ (READBACK);
  3. null      — 2 s of nothing: score < 0.05, no success;
  4. negative A— the SEED's plan (drop the pudding into the receptacle from above):
                 the chest is roofed — the drop never reaches any tray, no credit;
  5. negative B— the shut goal drawer is a dead end DIRECTLY: it is flush by cfg
                 arithmetic AND a 10 N press on its panel (solve's own force scale)
                 moves it < 1 mm — it already rests on its inner stop;
  6. negative C— wrong drawer: pudding dropped into the OPEN twin's tray, twin
                 pushed shut (which drives the goal drawer out through the beam) —
                 the chest LOOKS like "a drawer shut over the pudding" but it is the
                 wrong one: no load credit, no success, score <= the open latch;
  7. negative D— near miss: correct open + load, but the final push parks the goal
                 drawer ~2 cm short of flush — rejected, latched credit only;
  8. negative E— belly smuggle: pudding teleported INTO the chest cavity BEHIND the
                 open goal tray (not in it), goal pushed shut — the tail forks
                 bulldoze it to the back of the belly: never in the tray, no load
                 credit, no success;
  9. exactness — full gentle construction (solve's own phases) -> success() and
                 score == 1.0 (accept side of every boundary above);
 10. latch     — pushing the twin back in re-opens the loaded goal drawer: success
                 revoked, score falls to the latched 0.50 (earned credit stays);
 11. frames    — video frames recorded; saved as frames.npz in the CWD.

Run (forge): python -u -m simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i424.smoke --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--record_every", type=int, default=10)
parser.add_argument("--max_frames", type=int, default=280)
parser.add_argument("--out", type=str, default="frames.npz")  # CWD — the pipeline fetches it
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
# RTX recipe (proven on this render stack): kit mis-decodes the L20 driver version
# and silently rejects RTX -> the annotator returns EMPTY frames.
if not getattr(args, "kit_args", None):
    args.kit_args = "--/rtx/verifyDriverVersion/enabled=false"

app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robobench  # noqa: E402
from robobench.core import ENVS  # noqa: E402

robobench.discover()
try:
    from simgen_tasks.libero_kitchen_scene10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it_i424 import (  # noqa: F401,E501
        scene as scene_mod,
    )
except ImportError:  # standalone fallback (run from the package directory)
    import scene as scene_mod  # noqa: F401

# Watchdog: never leave a GPU zombie.
threading.Timer(1200.0, lambda: (print("SIM_GEN_SMOKE: TIMEOUT", flush=True),
                                 os._exit(3))).start()

# Same fingertip-scale drives as solve.py (shared numbers = shared honesty).
F_MAX = 10.0  # N on a drawer panel (closed-gripper push)
KP = 60.0
V_CAP = 0.10
K_APPROACH = 2.0
CUT_EXT = 0.004
HOVER = 0.049


def audit_geometry(c) -> None:
    """Fail-fast cfg arithmetic (no sim): the claims the task narrative rests on."""
    tol = 1e-6
    off2 = 2 * c.contact_offset + 0.0005  # summed contact offsets + margin
    # (a) shut drawer panel finishes flush in the front wall plane
    assert abs((c.closed_x + c.panel_c[0] - c.panel_s[0] / 2) - c.front_x[0]) < tol
    assert abs((c.closed_x + c.panel_c[0] + c.panel_s[0] / 2) - c.front_x[1]) < tol
    # (b) panel clears its aperture while sliding (sides / top / bottom)
    assert c.aperture_dy - c.panel_s[1] / 2 >= off2  # 4 mm sides
    panel_top = c.floor_z + c.panel_c[2] + c.panel_s[2] / 2
    panel_bot = c.floor_z + c.panel_c[2] - c.panel_s[2] / 2
    assert c.aperture_z_top - panel_top >= off2  # 5 mm top
    assert panel_bot - c.plinth_h >= off2  # 4 mm bottom
    # (c) anti-phase transmission: pin throw covers the full travel and the pin
    # stays captive in its fork slot (y overlap) at every angle
    assert abs(c.arm_len * math.sin(c.phi_max) - c.travel / 2) < tol
    fork_y = c.drawer_y + c.fork_dy
    for phi in (0.0, c.phi_max / 2, c.phi_max):
        pin_y = c.arm_len * math.cos(phi)
        assert abs(pin_y - fork_y) <= c.fork_s[1] / 2 - c.pin_r - 0.005
    # pin z-span covers the fork plates
    pin_lo = c.pivot[2] + c.pin_dz - c.pin_h / 2
    pin_hi = c.pivot[2] + c.pin_dz + c.pin_h / 2
    fork_lo = c.floor_z + c.fork_z - c.fork_s[2] / 2
    fork_hi = c.floor_z + c.fork_z + c.fork_s[2] / 2
    assert pin_lo <= fork_lo + 1e-6 and pin_hi >= fork_hi - 1e-6
    # (d) 4 mm fork backlash (> summed offsets: the transmission never wedges)
    gap = (c.fork_rear_x - c.fork_front_x - c.fork_s[0]) - 2 * c.pin_r
    assert gap >= off2 + 0.0005
    # (e) the chest is sealed against the pudding except via an open tray:
    # roofed (no vertical access), over-wall gap and under-floor gap both smaller
    # than the pudding, aperture sides too narrow beside the drawer body
    wall_top = c.floor_z + 0.0425 + c.traywall_s[2] / 2
    assert c.aperture_z_top - wall_top < c.cargo_size  # over the tray walls: 25 mm
    assert (c.floor_z - c.floor_size[2] / 2) - c.plinth_h < c.cargo_size  # under: 4 mm
    ap_side = c.aperture_dy - c.floor_size[1] / 2  # beside the drawer body: 16.5 mm
    assert ap_side < c.cargo_size
    assert c.roof_z[0] > wall_top  # roof clears the tallest drawer part it covers
    # (f) the OPEN tray exposes a landing longer than the pudding
    exposed = c.front_x[0] - (c.open_x + c.panel_c[0] + c.panel_s[0] / 2)
    assert exposed > c.cargo_size + 0.010
    # (g) tray band is honest: strictly inside the tray interior walls
    assert c.tray_x_lo > -0.080 - c.cargo_half and c.tray_x_hi < 0.070 + c.cargo_half
    assert c.tray_y_tol < 0.055
    print("[smoke] geometry audit: all cfg asserts hold", flush=True)


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ENVS.get("simgen.rocker_twin_drawers")().build(num_envs=args.num_envs, device=device)
    scene = env.scene
    c = scene.cfg
    n = env.num_envs
    no_action = torch.empty(0, device=device)
    all_ids = torch.arange(n, device=device)
    audit_geometry(c)

    # --- recording (viewport rgb annotator, the proven recipe) ---
    frames: list[np.ndarray] = []
    annot = None
    try:
        import omni.replicator.core as rep

        env.sim.set_render_mode(env.sim.RenderMode.PARTIAL_RENDERING)
        o = env.iscene.env_origins[0].detach().cpu().numpy().astype(float)
        env.sim.set_camera_view(tuple(np.array((-0.42, -0.62, 0.62)) + o),
                                tuple(np.array((0.42, 0.00, 0.08)) + o),
                                camera_prim_path="/OmniverseKit_Persp")
        rp = rep.create.render_product("/OmniverseKit_Persp", (800, 500))
        annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annot.attach([rp])
        for _ in range(6):
            env.sim.render()
        warm = np.asarray(annot.get_data())
        print(f"[smoke] camera ready, warmup frame shape={warm.shape}", flush=True)
        if warm.size == 0:
            print("[smoke] WARNING: annotator returns EMPTY frames — check the RTX recipe",
                  flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] camera setup FAILED ({exc!r}) — continuing without video", flush=True)

    step_i = 0

    def step(k: int) -> None:
        nonlocal step_i
        for _ in range(k):
            env.step(no_action, render=annot is not None)
            if annot is not None and step_i % args.record_every == 0 \
                    and len(frames) < args.max_frames:
                for _f in range(3):  # flush accumulated history (ghosting fix)
                    env.sim.render()
                arr = np.asarray(annot.get_data())
                if arr.size:
                    frames.append(arr[..., :3].astype(np.uint8).copy())
            step_i += 1

    def report(tag: str) -> None:
        e = scene.ext()[0].tolist()
        loc = scene.cargo_local()[0].tolist()
        print(f"[smoke] {tag:14s} | ext=(L {e[0]:+.4f}, R {e[1]:+.4f}) "
              f"cargo_local=({loc[0]:+.3f},{loc[1]:+.3f},{loc[2]:+.3f}) "
              f"in_tray={bool(scene.in_tray()[0])} "
              f"score={float(scene.score()[0]):.3f} success={bool(scene.success()[0])} "
              f"frames={len(frames)}", flush=True)

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, bool(cond)))
        print(f"[smoke] {'PASS' if cond else 'FAIL'}: {name}", flush=True)

    def settle_until(pred, max_steps: int = 480, poll: int = 10) -> bool:
        if pred():
            return True
        waited = 0
        while waited < max_steps:
            step(poll)
            waited += poll
            if pred():
                return True
        return False

    def servo_drawer(idx: int, cut_ext: float = CUT_EXT, budget: int = 3000) -> bool:
        """solve.py's velocity-servo: push drawer `idx` INWARD (+x) until its
        extension reaches `cut_ext`, then cut the drive."""
        body = scene.drawer_l if idx == 0 else scene.drawer_r
        for _ in range(budget):
            e = float(scene.ext()[0, idx])
            if e <= cut_ext:
                scene.drawer_drive[0, idx] = 0.0
                return True
            v = float(body.data.root_lin_vel_w[0, 0])
            # Ramp toward e=0 (solve.py's law): guarantees e crosses cut_ext with
            # margin. Ramping toward cut_ext itself asymptotes and never arrives.
            v_des = min(V_CAP, K_APPROACH * e)
            scene.drawer_drive[0, idx] = max(-F_MAX, min(F_MAX, KP * (v_des - v)))
            step(1)
        scene.drawer_drive[0, idx] = 0.0
        return False

    def teleport_cargo(pos: tuple, mark: bool = True) -> None:
        """Transport-only: pudding to a free-space world point, zero velocity, yaw 0."""
        st = torch.zeros(n, 13, device=device)
        st[:, 0:3] = scene.env_origins + torch.tensor(pos, device=device)
        st[:, 3] = 1.0
        scene.cargo.write_root_state_to_sim(st, all_ids)
        step(2)
        if mark:
            scene.mark_cargo_ref()  # wrench-frame reference for any later probe force

    def sides() -> tuple[int, int, float]:
        """(goal_idx, twin_idx, y_goal) for env 0."""
        side = int(scene.target_side[0])
        return side, 1 - side, (c.drawer_y if side == 0 else -c.drawer_y)

    def load_gently() -> bool:
        """solve's PHASE 1+2: open the goal via the twin, drop the pudding in."""
        goal_idx, twin_idx, y_goal = sides()
        ok_open = servo_drawer(twin_idx) and settle_until(
            lambda: float(scene.target_ext()[0]) >= c.open_thresh
            and bool(scene.settled()[0]), max_steps=360)
        teleport_cargo((c.open_x + c.drop_dx, y_goal,
                        c.floor_z + c.tray_rest_local_z + HOVER))
        ok_seat = settle_until(
            lambda: bool((scene.in_tray()
                          & (scene.cargo.data.root_lin_vel_w.norm(dim=-1)
                             < c.settle_cargo))[0]))
        return ok_open and ok_seat

    # ========================= 1. settle / clean-slate ========================================
    torch.manual_seed(3)
    env.reset()
    step(60)
    report("settled")
    finite = all(bool(torch.isfinite(b.data.root_state_w).all())
                 for b in (scene.drawer_l, scene.drawer_r, scene.rocker, scene.cargo)) \
        and bool(torch.isfinite(scene.score()).all())
    ext_ok = float(scene.target_ext()[0]) < c.closed_tol \
        and float(scene.twin_ext()[0]) > c.travel - 0.010
    p = (scene.cargo.data.root_pos_w[0] - scene.env_origins[0]).tolist()
    cargo_ok = c.cargo_x[0] - 0.02 <= p[0] <= c.cargo_x[1] + 0.02 \
        and abs(p[1]) <= c.cargo_y_half + 0.02 and abs(p[2] - c.cargo_half) < 0.01
    check("settle: clean reset (finite, goal shut / twin out, pudding on floor, score ~0)",
          finite and ext_ok and cargo_ok and float(scene.score()[0]) < 0.05)

    # ========================= 2. randomization readback ======================================
    draws = []
    for seed in (11, 12, 13, 14):
        torch.manual_seed(seed)
        env.reset()
        step(10)
        pud = (scene.cargo.data.root_pos_w[0, :2] - scene.env_origins[0, :2]).tolist()
        draws.append((int(scene.target_side[0]), round(pud[0], 3), round(pud[1], 3),
                      round(float(scene.target_ext()[0]), 4),
                      round(float(scene.twin_ext()[0]), 4)))
    print(f"[smoke] draws (side, cargo xy, goal/twin ext): {draws}", flush=True)
    xy = [(d[1], d[2]) for d in draws]
    max_pair = max(math.hypot(a[0] - b[0], a[1] - b[1])
                   for i, a in enumerate(xy) for b in xy[i + 1:])
    check("randomization is real (goal side flips across seeds; cargo xy differ; "
          "posed consistently)",
          len({d[0] for d in draws}) == 2 and max_pair > 0.03
          and all(d[3] < 0.010 and d[4] > c.travel - 0.012 for d in draws))

    # ========================= 3. null policy =================================================
    torch.manual_seed(3)
    env.reset()
    step(240)  # 2 s of nothing
    report("null")
    check("null policy: score < 0.05 and no success",
          float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 4. negative A: the SEED's plan (drop from above) ===============
    # The seed's move is drop-the-pudding-in-from-above. Here the chest is roofed and
    # the goal tray is shut inside it: the drop lands on the roof (or bounces to the
    # floor) and no tray is ever reached. Judged on the settled aftermath.
    torch.manual_seed(31)
    env.reset()
    step(30)
    goal_idx, twin_idx, y_goal = sides()
    teleport_cargo((c.closed_x, y_goal, 0.30))
    step(360)
    pz = float((scene.cargo.data.root_pos_w[0] - scene.env_origins[0])[2])
    report("roof-drop")
    check("negative (seed strategy): roofed chest — dropped pudding never reaches a tray",
          not bool(scene.in_tray()[0]) and float(scene.load_latch[0]) == 0.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) < 0.05
          and (abs(pz - (c.roof_z[1] + c.cargo_half)) < 0.02 or pz < 0.10))

    # ========================= 5. negative B: the shut goal drawer is a dead end ==============
    # Flush by construction (audited above + readback here), and pressing its panel
    # with solve's full 10 N does nothing — it already rests on its inner stop. (An
    # outward PULL would move it, but the flush knobless panel offers no purchase —
    # the embodiment argument in TASK.md; the press is the only force a fingertip
    # can deliver here.)
    torch.manual_seed(41)
    env.reset()
    step(30)
    goal_idx, twin_idx, y_goal = sides()
    e0 = float(scene.target_ext()[0])
    scene.drawer_drive[0, goal_idx] = F_MAX  # +x = inward press
    step(180)  # 1.5 s
    scene.drawer_drive[0, goal_idx] = 0.0
    step(60)
    e1 = float(scene.target_ext()[0])
    report("press-shut")
    check("negative (direct actuation): 10 N press on the shut goal panel moves it "
          "< 1 mm and earns nothing",
          e0 < 0.002 and abs(e1 - e0) < 0.001 and float(scene.open_latch[0]) == 0.0
          and float(scene.score()[0]) < 0.05 and not bool(scene.success()[0]))

    # ========================= 6. negative C: wrong drawer ====================================
    # Drop the pudding into the OPEN twin's tray (its tray is exposed from reset) and
    # push the twin shut. End state LOOKS like the seed's goal — "a drawer shut with
    # the pudding inside" — but it is the WRONG drawer, and shutting it drove the
    # goal drawer out through the beam. Only the open latch may fire.
    torch.manual_seed(51)
    env.reset()
    step(30)
    goal_idx, twin_idx, y_goal = sides()
    teleport_cargo((c.open_x + c.drop_dx, -y_goal,
                    c.floor_z + c.tray_rest_local_z + HOVER))
    step(240)  # land + settle in the twin tray
    twin_root = (scene.drawer_l if twin_idx == 0 else scene.drawer_r).data.root_pos_w[0]
    rel = (scene.cargo.data.root_pos_w[0] - twin_root).tolist()
    in_twin = c.tray_x_lo < rel[0] < c.tray_x_hi and abs(rel[1]) < c.tray_y_tol \
        and c.tray_z_lo < rel[2] < c.tray_z_hi
    ok_shut = servo_drawer(twin_idx)
    step(240)
    report("wrong-drawer")
    check("negative (wrong drawer): pudding shut inside the TWIN — no load credit, "
          "no success, only the open latch",
          in_twin and ok_shut and float(scene.twin_ext()[0]) < c.closed_tol
          and not bool(scene.in_tray()[0]) and float(scene.load_latch[0]) == 0.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.16)

    # ========================= 7. negative D: near miss (goal left ajar) ======================
    torch.manual_seed(61)
    env.reset()
    step(30)
    goal_idx, twin_idx, y_goal = sides()
    ok_load = load_gently()
    ok_park = servo_drawer(goal_idx, cut_ext=0.020)
    step(240)
    e_park = float(scene.target_ext()[0])
    report("ajar")
    check("negative (near miss): loaded but goal drawer parked ~2 cm short of flush "
          "— rejected, latched 0.50 only",
          ok_load and ok_park and c.closed_tol + 0.004 < e_park < 0.045
          and bool(scene.in_tray()[0]) and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.50) < 0.02)

    # ========================= 8. negative E: belly smuggle ===================================
    # Open the goal correctly, but teleport the pudding into the chest BELLY behind
    # the open tray (free plinth surface, not in any tray), then push the goal shut:
    # the tail forks bulldoze the pudding to the chest's back wall — it ends inside
    # the chest but never in a tray. No load credit, no success.
    torch.manual_seed(71)
    env.reset()
    step(30)
    goal_idx, twin_idx, y_goal = sides()
    ok_open = servo_drawer(twin_idx) and settle_until(
        lambda: float(scene.target_ext()[0]) >= c.open_thresh
        and bool(scene.settled()[0]), max_steps=360)
    teleport_cargo((0.455, y_goal, c.plinth_h + c.cargo_half + 0.004))
    step(120)
    ok_shut = servo_drawer(goal_idx, budget=2000)
    step(240)
    loc_x = float(scene.cargo_local()[0, 0])
    report("smuggled")
    check("negative (belly smuggle): pudding bulldozed behind the tray — never IN it, "
          "no load credit, no success",
          ok_open and loc_x > c.tray_x_hi + 0.02
          and not bool(scene.in_tray()[0]) and float(scene.load_latch[0]) == 0.0
          and not bool(scene.success()[0]) and float(scene.score()[0]) <= 0.16)
    print(f"[smoke]   smuggle aftermath: goal shut={ok_shut} "
          f"ext={float(scene.target_ext()[0]):.4f} cargo_local_x={loc_x:+.3f}", flush=True)

    # ========================= 9. exactness: success == score 1.0 =============================
    torch.manual_seed(91)
    env.reset()
    step(30)
    goal_idx, twin_idx, y_goal = sides()
    ok_load = load_gently()
    ok_shut = servo_drawer(goal_idx)
    ok = settle_until(lambda: bool(scene.success()[0]), max_steps=600)
    report("goal-state")
    check("exactness: full gentle construction -> success() and score == 1.0",
          ok_load and ok_shut and ok and abs(float(scene.score()[0]) - 1.0) < 1e-3)

    # ========================= 10. achievement latch ==========================================
    # Push the twin back in: the beam drives the LOADED goal drawer out again. The
    # live success term dies; the earned latches stay.
    reopened = servo_drawer(twin_idx, budget=2000)
    step(240)
    report("reopened")
    check("achievement latch: re-opening the loaded goal drawer revokes success; "
          "latched 0.50 remains",
          reopened and float(scene.target_ext()[0]) > c.open_thresh
          and not bool(scene.success()[0])
          and abs(float(scene.score()[0]) - 0.50) < 0.02)

    # ========================= 11. save + verdict =============================================
    if frames:
        arr = np.stack(frames, axis=0)
        np.savez_compressed(args.out, frames=arr, env="simgen.rocker_twin_drawers")
        print(f"[smoke] saved {arr.shape} -> {os.path.abspath(args.out)}", flush=True)
    check("video frames recorded", len(frames) >= 20)

    n_pass = sum(ok for _nm, ok in checks)
    all_ok = n_pass == len(checks)
    if all_ok:
        print(f"SIM_GEN_SMOKE: ALL PASS {n_pass}/{len(checks)}", flush=True)
    else:
        bad = [nm for nm, ok in checks if not ok]
        print(f"SIM_GEN_SMOKE: FAIL {n_pass}/{len(checks)} — failing: {bad}", flush=True)
    # Kit teardown hangs are routine: watchdog then hard exit.
    threading.Timer(10.0, lambda: os._exit(0 if all_ok else 1)).start()
    try:
        env.close()
        app.close()
    finally:
        os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
